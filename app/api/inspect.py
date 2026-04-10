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
    save_session_to_db, create_temp_session, get_latest_session_by_date_hour, get_latest_session_with_results
)
from app.models.database import SessionHistoryResponse
from app.services.auth_service import get_current_user
from uuid import UUID
import json

router = APIRouter()

@router.post("/api/inspect")
async def inspect_property(
    request: Request,
    files: list[UploadFile] = File(...),
):
    # Get current user from JWT token
    current_user = await get_current_user(request)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
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

    # Handle session creation or use existing
    db_session_id = None
    session_saved_to_db = False
    
    if incoming_session_id:
        try:
            db_session_id = UUID(incoming_session_id)
            # Check if session exists in DB, if not create it
            existing_session = await get_session(db_session_id)
            if not existing_session:
                await save_session_to_db(db_session_id)
            session_saved_to_db = True
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid session_id format")
    else:
        # Create new session and save to DB immediately
        new_session = await create_session(user_id=current_user.id)
        db_session_id = new_session.id
        incoming_session_id = str(new_session.id)
        session_saved_to_db = True

    storage_items = []
    image_records = []
    all_answers_by_image = []
    raw_analyses = []

    for file in files:
        # Read image so graph state remains serializable.
        image_bytes = await file.read()
        mime_type = file.content_type
        storage_info = None
        image_record = None

        try:
            storage_info = upload_inspection_image(
                image_bytes,
                mime_type,
                file.filename,
                incoming_session_id,
            )

            if storage_info:
                image_record = await create_image(
                    session_id=db_session_id,
                    image_url=storage_info.get("download_url", ""),
                    user_id=current_user.id
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
            if "API quota exceeded" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower() or "limit reached" in str(e).lower() or "429" in str(e):
                raise HTTPException(
                    status_code=429,
                    detail=str(e)
                ) from e
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error during property inspection: {str(e)}"
                ) from e

        raw_answers = result.get("answers", "")
        raw_analyses.append(raw_answers)

        if "```json" in raw_answers:
            start = raw_answers.find("```json") + 7
            end = raw_answers.find("```", start)
            if end > start:
                json_str = raw_answers[start:end].strip()
            else:
                json_str = raw_answers.replace("```json", "").replace("```", "").strip()
        else:
            json_str = raw_answers.strip()

        per_image_answers = ["No answer available"] * len(questions)
        try:
            answers_data = json.loads(json_str)
            answers_list = answers_data.get("answers", [])
            for i, answer in enumerate(answers_list[: len(questions)]):
                per_image_answers[i] = answer if answer else "No answer available"
        except Exception:
            pass

        all_answers_by_image.append(per_image_answers)

        if not image_record:
            image_record = await create_image(
                session_id=db_session_id,
                image_url=(storage_info or {}).get("download_url", ""),
                user_id=current_user.id
            )
        image_records.append(image_record)

    # Merge per-image answers by preferring the first meaningful answer.
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

    # Save inspection result to database
    inspection_result = {
        "question_answers": qa_pairs,
        "raw_analysis": "\n\n".join(raw_analyses)
    }
    
    # Save session to database only after successful analysis
    if not session_saved_to_db:
        await save_session_to_db(db_session_id)
        session_saved_to_db = True
    
    for image_record in image_records:
        await create_inspection_result(
            session_id=db_session_id,
            image_id=image_record.id,
            results=inspection_result,
            user_id=current_user.id
        )

    print(f"Final QA pairs: {qa_pairs}")
    
    return {
        "session_id": incoming_session_id,
        "question_answers": qa_pairs,
        "storage": storage_items[0] if storage_items else None,
        "storage_items": storage_items,
    }


@router.get("/api/sessions/latest")
async def get_latest_session_endpoint():
    """Get the latest session overall"""
    session = await get_latest_session_by_date_hour()
    if not session:
        raise HTTPException(status_code=404, detail="No session found")
    return session


@router.get("/api/sessions/latest-with-results")
async def get_latest_session_with_results_endpoint():
    """Get the latest session that has inspection results"""
    try:
        session = await get_latest_session_with_results()
        if not session:
            raise HTTPException(status_code=404, detail="No session with results found")
        return session
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in latest session with results endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching session with results: {str(e)}")


@router.get("/api/sessions/with-results")
async def get_sessions_with_results_endpoint():
    """Get all sessions that have inspection results"""
    try:
        from app.services.database_service import async_session, select, Session, InspectionResult
        
        async with async_session() as session:
            # Get all session IDs that have inspection results
            result = await session.execute(
                select(InspectionResult.session_id).distinct()
            )
            session_ids = [row[0] for row in result.fetchall()]
            
            if session_ids:
                # Get the sessions
                sessions_result = await session.execute(
                    select(Session).where(Session.id.in_(session_ids))
                    .order_by(Session.created_at.desc())
                )
                sessions = sessions_result.scalars().all()
                
                return {"sessions": [{"id": str(s.id), "created_at": s.created_at.isoformat()} for s in sessions]}
            
            return {"sessions": []}
    except Exception as e:
        print(f"Error in sessions with results endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/sessions")
async def create_new_session(request: Request):
    """Create a new session"""
    # Get current user from JWT token
    current_user = await get_current_user(request)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    session = await create_session(user_id=current_user.id)
    return session


@router.get("/api/sessions/{session_id}")
async def get_session_history_endpoint(session_id: str, request: Request):
    """Get complete session history"""
    # Get current user from JWT token
    current_user = await get_current_user(request)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        session_uuid = UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session_id format")
    
    history = await get_session_history(session_uuid)
    if not history:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return history


@router.post("/api/sessions/{session_id}/conversations")
async def add_conversation(session_id: str, role: str, message: str, request: Request):
    """Add a conversation message to a session"""
    # Get current user from JWT token
    current_user = await get_current_user(request)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        session_uuid = UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session_id format")
    
    if role not in ["user", "ai"]:
        raise HTTPException(status_code=400, detail="Role must be 'user' or 'ai'")
    
    # Check if session exists
    session = await get_session(session_uuid)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    conversation = await create_conversation(session_uuid, role, message, current_user.id)
    return conversation