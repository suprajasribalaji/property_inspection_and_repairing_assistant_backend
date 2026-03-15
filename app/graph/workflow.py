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

    image_bytes = state["image_bytes"]
    mime_type = state["mime_type"]

    observations = await observe_property_image(image_bytes, mime_type)

    return {"observations": observations}

async def inspection_node(state: InspectionState):

    image_bytes = state["image_bytes"]
    mime_type = state["mime_type"]
    questions = state["questions"]
    observations = state["observations"]

    answers = await analyze_property_image(observations, image_bytes, mime_type, questions)

    return {"answers": answers}


def build_graph():

    workflow = StateGraph(InspectionState)

    workflow.add_node("observation", observation_node)

    workflow.add_node("inspection", inspection_node)

    workflow.set_entry_point("observation")

    workflow.add_edge("observation", "inspection")

    workflow.set_finish_point("inspection")

    return workflow.compile()


graph = build_graph()


async def run_inspection_graph(image_bytes: bytes, mime_type: str, questions: list[str]):

    result = await graph.ainvoke({
        "image_bytes": image_bytes,
        "mime_type": mime_type,
        "questions": questions
    })

    return result.get("answers", "") 