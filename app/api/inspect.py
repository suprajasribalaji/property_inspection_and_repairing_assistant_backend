from fastapi import APIRouter, UploadFile, File
from app.services.pdf_loader import load_predefined_questions
from app.graph.workflow import run_inspection_graph

router = APIRouter()

@router.post("/api/inspect")
async def inspect_property(file: UploadFile = File(...)):
    questions = load_predefined_questions()
    
    # Read image here so the graph state is serializable
    image_bytes = await file.read()
    mime_type = file.content_type

    result = await run_inspection_graph(image_bytes, mime_type, questions)
    
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
        import json
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

    print(f"Final QA pairs: {qa_pairs}")
    
    return {
        "question_answers": qa_pairs
    }