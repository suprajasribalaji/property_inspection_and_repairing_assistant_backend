from fastapi import APIRouter, UploadFile, File, HTTPException
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
from uuid import UUID
import json

router = APIRouter()

@router.post("/api/inspect")
async def inspect_property(file: UploadFile = File(...), session_id: str = None):
    questions = load_predefined_questions()
    
    # Read image here so the graph state is serializable
    image_bytes = await file.read()
    mime_type = file.content_type

    # Handle session creation or use existing
    db_session_id = None
    session_saved_to_db = False
    
    if session_id:
        try:
            db_session_id = UUID(session_id)
            # Check if session exists in DB, if not create it
            existing_session = await get_session(db_session_id)
            if not existing_session:
                await save_session_to_db(db_session_id)
            session_saved_to_db = True
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid session_id format")
    else:
        # Create new session and save to DB immediately
        new_session = await create_session()
        db_session_id = new_session.id
        session_id = str(new_session.id)
        session_saved_to_db = True

    storage_info = None
    image_record = None
    
    # Upload to Firebase and save to database
    try:
        storage_info = upload_inspection_image(
            image_bytes,
            mime_type,
            file.filename,
            session_id,
        )
        
        if storage_info:
            # Save image record to database
            image_record = await create_image(
                session_id=db_session_id,
                image_url=storage_info.get("download_url", "")
            )
    except Exception as e:
        if is_firebase_configured():
            raise HTTPException(
                status_code=500,
                detail=f"Failed to upload image to storage: {e!s}",
            ) from e

    try:
        result = await run_inspection_graph(image_bytes, mime_type, questions)
    except Exception as e:
        if "API quota exceeded" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            # Handle quota exceeded gracefully
            fallback_result = {
                "observations": ["API quota exceeded - unable to analyze image"],
                "answers": json.dumps({
                    "answers": ["API quota exceeded. Please try again later or upgrade your plan."] * len(questions)
                })
            }
            result = fallback_result
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Error during property inspection: {str(e)}"
            ) from e
    
    # Parse the answers and clean up JSON
    qa_pairs = []
    raw_answers = result.get("answers", "")
    
    # Extract JSON from markdown if present
    if "```json" in raw_answers:
        start = raw_answers.find("```json") + 7
        end = raw_answers.find("```", start)
        if end > start:
            json_str = raw_answers[start:end].strip()
        else:
            json_str = raw_answers.replace("```json", "").replace("```", "").strip()
    else:
        json_str = raw_answers.strip()
    
    # Parse JSON
    try:
        answers_data = json.loads(json_str)
        answers_list = answers_data.get("answers", [])
        
        # Create question-answer pairs
        for i, (question, answer) in enumerate(zip(questions, answers_list)):
            qa_pairs.append({
                "question": question,
                "answer": answer if answer else "No answer available"
            })
        
        # If we have more questions than answers, add remaining questions
        if len(answers_list) < len(questions):
            for i in range(len(answers_list), len(questions)):
                qa_pairs.append({
                    "question": questions[i],
                    "answer": "No answer available"
                })
    except Exception as e:
        # Fallback: create pairs with "No answer available"
        for question in questions:
            qa_pairs.append({
                "question": question,
                "answer": "No answer available"
            })

    # Save inspection result to database
    inspection_result = {
        "question_answers": qa_pairs,
        "raw_analysis": raw_answers
    }
    
    # Save session to database only after successful analysis
    if not session_saved_to_db:
        await save_session_to_db(db_session_id)
        session_saved_to_db = True
    
    if image_record:
        await create_inspection_result(
            session_id=db_session_id,
            image_id=image_record.id,
            results=inspection_result
        )

    print(f"Final QA pairs: {qa_pairs}")
    
    return {
        "session_id": session_id,
        "question_answers": qa_pairs,
        "storage": storage_info,
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
async def create_new_session():
    """Create a new session"""
    session = await create_session()
    return session


@router.get("/api/sessions/{session_id}")
async def get_session_history_endpoint(session_id: str):
    """Get complete session history"""
    try:
        session_uuid = UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session_id format")
    
    history = await get_session_history(session_uuid)
    if not history:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return history


@router.get("/api/sessions")
async def get_all_sessions_endpoint():
    """Get all sessions"""
    sessions = await get_all_sessions()
    return {"sessions": sessions}


@router.post("/api/sessions/{session_id}/conversations")
async def add_conversation(session_id: str, role: str, message: str):
    """Add a conversation message to a session"""
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
    
    conversation = await create_conversation(session_uuid, role, message)
    return conversation