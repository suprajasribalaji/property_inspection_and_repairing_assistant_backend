from __future__ import annotations

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Any
from uuid import UUID

from app.services.database_service import (
    get_latest_session_with_results,
    get_session_history,
    create_conversation,
)
from app.graph.chat_graph import chat_graph
from app.models.database import User
from app.routes.auth import get_current_user

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


def _build_fast_fallback_answer(question: str, findings: list[dict[str, Any]]) -> str:
    top = findings[:5]
    bullets = "\n".join(
        [f"- {item.get('question', '')}: {item.get('answer', '')}" for item in top]
    )
    return (
        "I could not complete a full AI response right now, but here are the most relevant findings:\n"
        f"{bullets}\n\nPlease retry your question in a moment."
    )


@router.post("/chat")
async def chat_assistant(
    req: ChatRequest,
    current_user: User = Depends(get_current_user)          # ← JWT protection
):
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Scope session lookup to current user
    session = await get_latest_session_with_results(user_id=current_user.id)
    if not session:
        raise HTTPException(status_code=404, detail="No inspection session found. Please run an inspection first.")

    session_id: UUID = session.id

    history = await get_session_history(session_id)
    if not history:
        raise HTTPException(status_code=404, detail="Session history not found")

    # Verify ownership
    if history.session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied to this session")

    # Merge findings from ALL inspection results (not just latest)
    findings: list[dict] = []
    for inspection_result in history.inspection_results:
        results_dict = inspection_result.results or {}
        findings.extend(results_dict.get("question_answers", []) or [])

    findings = _filter_valid_findings(findings)

    if not findings:
        return {
            "answer": (
                "I could not find enough visible inspection findings to answer reliably. "
                "Please upload additional close-up photos by area, then ask again."
            )
        }

    conversation: list[dict[str, str]] = []
    for conv in (history.conversations or [])[-10:]:
        role = "user" if conv.role == "user" else "assistant"
        conversation.append({"role": role, "message": conv.message})

    findings = findings[:30]

    try:
        agent_result = await chat_graph.ainvoke({
            "question": question,
            "findings": findings,
            "conversation": conversation,
        })
        assistant_response = agent_result.get("assistant_response") or ""
        if not assistant_response.strip():
            assistant_response = "I can help, but I need a bit more detail to answer accurately."
    except Exception as e:
        print(f"Chat agent failed: {e}")
        assistant_response = _build_fast_fallback_answer(question, findings)

    await create_conversation(session_id, "user", question)
    await create_conversation(session_id, "ai", assistant_response)

    return {"answer": assistant_response}