from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Depends
from app.services.pdf_loader import load_predefined_questions
from app.graph.workflow import run_inspection_graph
from app.services.firebase_storage_service import (
    upload_inspection_image,
    is_firebase_configured,
)
from app.services.database_service import (
    create_session, create_image, create_inspection_result,
    get_session_history, get_all_sessions, get_session,
    save_session_to_db, 
    get_latest_session_by_date_hour, 
    get_latest_session_with_results
)
from app.models.database import User, SessionHistoryResponse
from app.routes.auth import get_current_user
from uuid import UUID
import json

router = APIRouter()


@router.post("/api/inspect")
async def inspect_property(
    request: Request,
    files: list[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),         # ← JWT protection
):
    questions = load_predefined_questions()
    incoming_session_id = request.query_params.get("session_id")
    if not incoming_session_id:
        try:
            form = await request.form()
            incoming_session_id = form.get("session_id")
        except Exception:
            incoming_session_id = None

    if not files:
        raise HTTPException(status_code=400, detail="At least one image is required")

    db_session_id = None
    session_saved_to_db = False

    if incoming_session_id:
        try:
            db_session_id = UUID(incoming_session_id)
            existing_session = await get_session(db_session_id)
            if not existing_session:
                await save_session_to_db(db_session_id, user_id=current_user.id)  # ← pass user_id
            else:
                # Verify session belongs to current user
                if existing_session.user_id != current_user.id:
                    raise HTTPException(status_code=403, detail="Access denied to this session")
            session_saved_to_db = True
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid session_id format")
    else:
        new_session = await create_session(user_id=current_user.id)   # ← pass user_id
        db_session_id = new_session.id
        incoming_session_id = str(new_session.id)
        session_saved_to_db = True

    storage_items = []
    image_records = []
    all_answers_by_image = []
    raw_analyses = []

    for file in files:
        image_bytes = await file.read()
        mime_type = file.content_type
        storage_info = None
        image_record = None

        try:
            storage_info = upload_inspection_image(
                image_bytes, mime_type,
                file.filename, incoming_session_id,
            )
            if storage_info:
                image_record = await create_image(
                    session_id=db_session_id,
                    image_url=storage_info.get("download_url", "")
                )
                storage_items.append(storage_info)
        except Exception as e:
            if is_firebase_configured():
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to upload image to storage: {e!s}",
                ) from e

        try:
            result = await run_inspection_graph(image_bytes, mime_type, questions)
        except Exception as e:
            if "quota" in str(e).lower() or "RESOURCE_EXHAUSTED" in str(e) or "limit reached" in str(e).lower():
                raise HTTPException(status_code=429, detail="API quota exceeded. Please try again later.")
            raise HTTPException(status_code=500, detail=f"Error during inspection: {str(e)}")

        raw_answers = result.get("answers", "")
        raw_analyses.append(raw_answers)

        if "```json" in raw_answers:
            start = raw_answers.find("```json") + 7
            end = raw_answers.find("```", start)
            json_str = raw_answers[start:end].strip() if end > start else raw_answers.replace("```json", "").replace("```", "").strip()
        else:
            json_str = raw_answers.strip()

        per_image_answers = ["No answer available"] * len(questions)
        try:
            answers_data = json.loads(json_str)
            answers_list = answers_data.get("answers", [])
            for i, answer in enumerate(answers_list[:len(questions)]):
                per_image_answers[i] = answer if answer else "No answer available"
        except Exception:
            pass

        all_answers_by_image.append(per_image_answers)

        if not image_record:
            image_record = await create_image(
                session_id=db_session_id,
                image_url=(storage_info or {}).get("download_url", "")
            )
        image_records.append(image_record)

    # Merge answers across images
    invalid_answers = {
        "not visible in the image",
        "no answer available",
        "api quota exceeded. please try again later or upgrade your plan.",
    }
    qa_pairs = []
    for i, question in enumerate(questions):
        selected = "Not visible in the image"
        for answers in all_answers_by_image:
            candidate = (answers[i] or "").strip()
            if candidate and candidate.lower() not in invalid_answers:
                selected = candidate
                break
            if selected == "Not visible in the image" and candidate:
                selected = candidate
        qa_pairs.append({"question": question, "answer": selected})

    inspection_result = {
        "question_answers": qa_pairs,
        "raw_analysis": "\n\n".join(raw_analyses)
    }

    if not session_saved_to_db:
        await save_session_to_db(db_session_id, user_id=current_user.id)
        session_saved_to_db = True

    for image_record in image_records:
        await create_inspection_result(
            session_id=db_session_id,
            image_id=image_record.id,
            results=inspection_result
        )

    return {
        "session_id": incoming_session_id,
        "question_answers": qa_pairs,
        "storage": storage_items[0] if storage_items else None,
        "storage_items": storage_items,
    }

@router.post("/api/sessions")
async def create_new_session(
    current_user: User = Depends(get_current_user)   # ← add this
):
    session = await create_session(user_id=current_user.id)
    return session

@router.get("/api/sessions/latest")
async def get_latest_session_endpoint(
    current_user: User = Depends(get_current_user)          # ← JWT protection
):
    session = await get_latest_session_by_date_hour(user_id=current_user.id)
    if not session:
        raise HTTPException(status_code=404, detail="No session found")
    return session


@router.get("/api/sessions/latest-with-results")
async def get_latest_session_with_results_endpoint(
    current_user: User = Depends(get_current_user)          # ← JWT protection
):
    try:
        session = await get_latest_session_with_results(user_id=current_user.id)
        if not session:
            raise HTTPException(status_code=404, detail="No session with results found")
        return session
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching session: {str(e)}")


@router.get("/api/sessions/{session_id}")
async def get_session_history_endpoint(
    session_id: str,
    current_user: User = Depends(get_current_user)          # ← JWT protection
):
    try:
        session_uuid = UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session_id format")

    history = await get_session_history(session_uuid)
    if not history:
        raise HTTPException(status_code=404, detail="Session not found")

    # Ensure session belongs to current user
    if history.session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied to this session")

    return history


@router.get("/api/sessions")
async def get_all_sessions_endpoint(
    current_user: User = Depends(get_current_user)          # ← JWT protection
):
    sessions = await get_all_sessions(user_id=current_user.id)
    return {"sessions": sessions}
