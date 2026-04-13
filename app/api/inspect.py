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
    print("[API_INSPECT] ===== STARTING PROPERTY INSPECTION =====")
    questions = load_predefined_questions()
    print(f"[API_INSPECT] Loaded {len(questions)} questions")
    
    incoming_session_id = request.query_params.get("session_id")
    print(f"[API_INSPECT] Session ID from query: {incoming_session_id}")
    if not incoming_session_id:
        try:
            form = await request.form()
            incoming_session_id = form.get("session_id")
            print(f"[API_INSPECT] Session ID from form: {incoming_session_id}")
        except Exception as e:
            print(f"[API_INSPECT] Error getting session from form: {e}")
            incoming_session_id = None

    print(f"[API_INSPECT] Files received: {len(files)}")
    if not files:
        print("[API_INSPECT] No files provided - raising 400 error")
        raise HTTPException(status_code=400, detail="At least one image is required")

    db_session_id = None
    session_saved_to_db = False

    print(f"[API_INSPECT] Processing session: {incoming_session_id}")
    if incoming_session_id:
        try:
            db_session_id = UUID(incoming_session_id)
            existing_session = await get_session(db_session_id)
            print(f"[API_INSPECT] Existing session found: {existing_session is not None}")
            if not existing_session:
                print(f"[API_INSPECT] Creating new session record")
                await save_session_to_db(db_session_id, user_id=current_user.id)  # ← pass user_id
            else:
                # Verify session belongs to current user
                if existing_session.user_id != current_user.id:
                    print(f"[API_INSPECT] Access denied - session belongs to user {existing_session.user_id}, current user: {current_user.id}")
                    raise HTTPException(status_code=403, detail="Access denied to this session")
            session_saved_to_db = True
        except ValueError:
            print(f"[API_INSPECT] Invalid session ID format: {incoming_session_id}")
            raise HTTPException(status_code=400, detail="Invalid session_id format")
    else:
        print("[API_INSPECT] Creating new session")
        new_session = await create_session(user_id=current_user.id)   # ← pass user_id
        db_session_id = new_session.id
        incoming_session_id = str(new_session.id)
        session_saved_to_db = True
        print(f"[API_INSPECT] Created new session: {incoming_session_id}")

    storage_items = []
    image_records = []
    all_answers_by_image = []
    raw_analyses = []

    print(f"[API_INSPECT] Starting to process {len(files)} files")
    for i, file in enumerate(files):
        print(f"[API_INSPECT] --- Processing file {i+1}/{len(files)}: {file.filename} ---")
        image_bytes = await file.read()
        mime_type = file.content_type
        print(f"[API_INSPECT] File read: {len(image_bytes)} bytes, mime: {mime_type}")
        storage_info = None
        image_record = None

        try:
            print(f"[API_INSPECT] Attempting storage upload")
            storage_info = upload_inspection_image(
                image_bytes, mime_type,
                file.filename, incoming_session_id,
            )
            if storage_info:
                print(f"[API_INSPECT] Storage upload successful")
                image_record = await create_image(
                    session_id=db_session_id,
                    image_url=storage_info.get("download_url", "")
                )
                storage_items.append(storage_info)
                print(f"[API_INSPECT] Image record created: {image_record.id}")
            else:
                print(f"[API_INSPECT] Storage upload returned None")
        except Exception as e:
            print(f"[API_INSPECT] Storage upload error: {e}")
            if is_firebase_configured():
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to upload image to storage: {e!s}",
                ) from e
            else:
                print(f"[API_INSPECT] Firebase not configured, continuing")

        try:
            print(f"[API_INSPECT] Starting inspection graph for file {i+1}")
            result = await run_inspection_graph(image_bytes, mime_type, questions)
            print(f"[API_INSPECT] Inspection graph completed for file {i+1}")
        except Exception as e:
            print(f"[API_INSPECT] Inspection graph error for file {i+1}: {e}")
            if "quota" in str(e).lower() or "RESOURCE_EXHAUSTED" in str(e) or "limit reached" in str(e).lower():
                print(f"[API_INSPECT] Quota exceeded - raising 429")
                raise HTTPException(status_code=429, detail="API quota exceeded. Please try again later.")
            print(f"[API_INSPECT] General error - raising 500")
            raise HTTPException(status_code=500, detail=f"Error during inspection: {str(e)}")

        raw_answers = result.get("answers", "")
        print(f"[API_INSPECT] Got raw answers: {len(raw_answers)} chars")
        raw_analyses.append(raw_answers)

        if "```json" in raw_answers:
            start = raw_answers.find("```json") + 7
            end = raw_answers.find("```", start)
            json_str = raw_answers[start:end].strip() if end > start else raw_answers.replace("```json", "").replace("```", "").strip()
        else:
            json_str = raw_answers.strip()
        
        print(f"[API_INSPECT] Parsed JSON string: {len(json_str)} chars")

        per_image_answers = ["No answer available"] * len(questions)
        try:
            answers_data = json.loads(json_str)
            answers_list = answers_data.get("answers", [])
            print(f"[API_INSPECT] Parsed {len(answers_list)} answers")
            for i, answer in enumerate(answers_list[:len(questions)]):
                per_image_answers[i] = answer if answer else "No answer available"
        except Exception as e:
            print(f"[API_INSPECT] JSON parsing error: {e}")
            pass

        all_answers_by_image.append(per_image_answers)
        print(f"[API_INSPECT] Added answers for file {i+1} to collection")

        if not image_record:
            print(f"[API_INSPECT] Creating image record without storage")
            image_record = await create_image(
                session_id=db_session_id,
                image_url=(storage_info or {}).get("download_url", ""),
            )
        image_records.append(image_record)
        print(f"[API_INSPECT] --- Completed file {i+1} processing ---")

    # Merge answers across images
    print(f"[API_INSPECT] Starting answer merge from {len(all_answers_by_image)} images")
    invalid_answers = {
        "not visible in the image",
        "no answer available",
        "api quota exceeded. please try again later or upgrade your plan.",
    }
    qa_pairs = []
    valid_answers_count = 0
    for i, question in enumerate(questions):
        selected = "Not visible in the image"
        for answers in all_answers_by_image:
            candidate = (answers[i] or "").strip()
            if candidate and candidate.lower() not in invalid_answers:
                selected = candidate
                valid_answers_count += 1
                break
            if selected == "Not visible in the image" and candidate:
                selected = candidate
        qa_pairs.append({"question": question, "answer": selected})
    
    print(f"[API_INSPECT] Answer merge complete: {len(qa_pairs)} total pairs, {valid_answers_count} valid answers")

    inspection_result = {
        "question_answers": qa_pairs,
        "raw_analysis": "\n\n".join(raw_analyses)
    }
    print(f"[API_INSPECT] Created inspection result with {len(raw_analyses)} analyses")

    if not session_saved_to_db:
        print(f"[API_INSPECT] Saving session to database")
        await save_session_to_db(db_session_id, user_id=current_user.id)
        session_saved_to_db = True

    print(f"[API_INSPECT] Creating inspection results for {len(image_records)} images")
    for image_record in image_records:
        await create_inspection_result(
            session_id=db_session_id,
            image_id=image_record.id,
            results=inspection_result
        )

    response_data = {
        "session_id": incoming_session_id,
        "question_answers": qa_pairs,
        "storage": storage_items[0] if storage_items else None,
        "storage_items": storage_items,
    }
    print(f"[API_INSPECT] ===== PROPERTY INSPECTION COMPLETED =====")
    print(f"[API_INSPECT] Returning {len(qa_pairs)} Q&A pairs")
    return response_data

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