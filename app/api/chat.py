from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any
from uuid import UUID

from app.services.database_service import (
    get_latest_session_with_results,
    get_session_history,
    create_conversation,
)
from app.graph.chat_graph import chat_graph


router = APIRouter()


class ChatRequest(BaseModel):
    question: str


def _normalize_answer(s: str) -> str:
    return (s or "").strip().lower()


def _filter_valid_findings(findings: list[dict[str, Any]]) -> list[dict]:
    valid = []
    for item in findings or []:
        ans = _normalize_answer(item.get("answer", ""))
        if ans in {"not visible in the image", "no answer available"}:
            continue
        valid.append(item)
    return valid


@router.post("/chat")
async def chat_assistant(req: ChatRequest):
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    session = await get_latest_session_with_results()
    if not session:
        raise HTTPException(status_code=404, detail="No inspection session with results found")

    session_id: UUID = session.id

    history = await get_session_history(session_id)
    if not history:
        raise HTTPException(status_code=404, detail="Session history not found")

    # Use the most recent inspection result set.
    findings: list[dict] = []
    if history.inspection_results:
        latest = history.inspection_results[-1]
        results_dict = latest.results or {}
        findings = results_dict.get("question_answers", []) or []

    findings = _filter_valid_findings(findings)

    # Provide a small conversation window for follow-ups.
    conversation: list[dict[str, str]] = []
    for conv in (history.conversations or [])[-10:]:
        role = "user" if conv.role == "user" else "assistant"
        conversation.append({"role": role, "message": conv.message})

    agent_result = await chat_graph.ainvoke(
        {
            "question": question,
            "findings": findings,
            "conversation": conversation,
        }
    )

    assistant_response = agent_result.get("assistant_response") or ""
    if not assistant_response.strip():
        assistant_response = "I can help, but I need a bit more detail to answer accurately."

    # Persist conversation so refresh restores chat.
    await create_conversation(session_id, "user", question)
    await create_conversation(session_id, "ai", assistant_response)

    return {"answer": assistant_response}

