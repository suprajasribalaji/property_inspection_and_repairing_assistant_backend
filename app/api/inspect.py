from fastapi import APIRouter, UploadFile, File
from app.services.pdf_loader import load_predefined_questions
from app.graph.workflow import run_inspection

router = APIRouter()

@router.post("/inspect")
async def inspect_property(file: UploadFile = File(...)):
    questions = load_predefined_questions()
    
    # Read image here so the graph state is serializable
    image_bytes = await file.read()
    mime_type = file.content_type
    
    result = await run_inspection(image_bytes, mime_type, questions)
    return {"answers": result} 