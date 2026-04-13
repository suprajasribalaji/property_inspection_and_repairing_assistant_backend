from langgraph.graph import StateGraph
from typing import TypedDict, Annotated
import operator
from app.services.image_analysis_service import analyze_property_image, observe_property_image

class InspectionState(TypedDict):
    image_bytes: bytes
    mime_type: str
    questions: list[str]
    observations: list[str]
    answers: dict

async def observation_node(state: InspectionState):
    print("[GRAPH_OBSERVATION] Starting observation node")
    
    image_bytes = state["image_bytes"]
    mime_type = state["mime_type"]
    print(f"[GRAPH_OBSERVATION] Processing image - size: {len(image_bytes)} bytes, mime: {mime_type}")

    observations = await observe_property_image(image_bytes, mime_type)
    print(f"[GRAPH_OBSERVATION] Got {len(observations)} observations")

    return {"observations": observations}

async def inspection_node(state: InspectionState):
    print("[GRAPH_INSPECTION] Starting inspection node")
    
    image_bytes = state["image_bytes"]
    mime_type = state["mime_type"]
    questions = state["questions"]
    observations = state["observations"]
    print(f"[GRAPH_INSPECTION] Processing with {len(questions)} questions and {len(observations)} observations")

    answers = await analyze_property_image(observations, image_bytes, mime_type, questions)
    print(f"[GRAPH_INSPECTION] Got answers: {type(answers)}")

    return {"answers": answers}

def build_graph():
    print("[GRAPH_BUILD] Building inspection graph")
    
    workflow = StateGraph(InspectionState)

    print("[GRAPH_BUILD] Adding observation node")
    workflow.add_node("observation", observation_node)

    print("[GRAPH_BUILD] Adding inspection node")
    workflow.add_node("inspection", inspection_node)

    print("[GRAPH_BUILD] Setting entry point to observation")
    workflow.set_entry_point("observation")

    print("[GRAPH_BUILD] Adding edge: observation -> inspection")
    workflow.add_edge("observation", "inspection")

    print("[GRAPH_BUILD] Setting finish point to inspection")
    workflow.set_finish_point("inspection")

    compiled_graph = workflow.compile()
    print("[GRAPH_BUILD] Graph built and compiled successfully")
    return compiled_graph

graph = build_graph()

async def run_inspection_graph(image_bytes: bytes, mime_type: str, questions: list[str]):
    print(f"[GRAPH_RUN] Starting inspection graph execution")
    print(f"[GRAPH_RUN] Input: image_size={len(image_bytes)} bytes, mime_type={mime_type}, questions={len(questions)}")
    
    try:
        print("[GRAPH_RUN] Invoking graph")
        result = await graph.ainvoke({
            "image_bytes": image_bytes,
            "mime_type": mime_type,
            "questions": questions
        })
        print(f"[GRAPH_RUN] Graph execution completed")
        print(f"[GRAPH_RUN] Result keys: {list(result.keys()) if result else 'None'}")
        
        answers = result.get("answers", "")
        print(f"[GRAPH_RUN] Extracted answers: {type(answers)}")
        if isinstance(answers, dict):
            print(f"[GRAPH_RUN] Answers dict keys: {list(answers.keys())}")
        elif isinstance(answers, str):
            print(f"[GRAPH_RUN] Answers string length: {len(answers)}")
        
        return answers
    except Exception as e:
        print(f"[GRAPH_RUN] Error during graph execution: {e}")
        raise