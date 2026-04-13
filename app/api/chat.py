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
    conversation_history: list[dict] = []  # [{"role": "user"|"assistant", "message": str}]


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
    return (
        "I apologize, but I'm having trouble connecting to the AI services right now to process your question properly. "
        "Please try asking your question again in a few moments."
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

    print(f"[CHAT_API_DEBUG] Total findings before filtering: {len(findings)}")
    for i, f in enumerate(findings[:3]):  # Log first 3 findings
        print(f"[CHAT_API_DEBUG] Finding {i}: Q='{f.get('question', '')[:50]}...' A='{f.get('answer', '')[:50]}...'")
    
    findings = _filter_valid_findings(findings)
    print(f"[CHAT_API_DEBUG] Valid findings after filtering: {len(findings)}")

    if not findings:
        # Try to provide general guidance even without specific findings
        question_lower = question.lower()
        if any(keyword in question_lower for keyword in ["bathroom", "kitchen", "electrical", "plumbing", "structure"]):
            return {
                "answer": (
                    f"While I don't have specific findings from your photos yet, I can provide general guidance for {question_lower.split()[0]} areas. "
                    "For more accurate advice, please upload close-up photos of the specific areas you're concerned about. "
                    "In the meantime, what specific aspect would you like general guidance on?"
                )
            }
        else:
            return {
                "answer": (
                    "I don't have enough specific findings from your photos to provide detailed guidance. "
                    "Please upload close-up photos of the areas you're concerned about, and I'll be able to give you more targeted advice. "
                    "What specific areas or issues would you like help with?"
                )
            }

    # Build conversation: prefer client-sent history (always current), supplement with DB if client sends nothing
    # Client sends: [{"role": "user"|"assistant", "message": str}]
    client_history = [
        {"role": msg.get("role", "user"), "message": msg.get("message", "")}
        for msg in (req.conversation_history or [])
        if msg.get("message", "").strip()
    ]

    if client_history:
        # Client already has the full in-memory conversation — use it directly
        conversation = client_history[-20:]  # keep last 20 turns
        print(f"[CHAT_API] Using client-sent conversation history: {len(conversation)} turns")
    else:
        # Fallback: reconstruct from DB (e.g. on page refresh)
        conversation = []
        for conv in (history.conversations or [])[-20:]:
            role = "user" if conv.role == "user" else "assistant"
            conversation.append({"role": role, "message": conv.message})
        print(f"[CHAT_API] Using DB conversation history: {len(conversation)} turns")

    findings = findings[:50]  # allow more findings for richer context

    try:
        print(f"[CHAT_API] Invoking chat graph with {len(findings)} findings")
        agent_result = await chat_graph.ainvoke({
            "question": question,
            "findings": findings,
            "conversation": conversation,
        })
        assistant_response = agent_result.get("assistant_response") or ""
        print(f"[CHAT_API] Chat graph response: {assistant_response[:100]}...")
        if not assistant_response.strip():
            assistant_response = "I can help, but I need a bit more detail to answer accurately."
    except Exception as e:
        print(f"[CHAT_API] Chat agent failed: {e}")
        print(f"[CHAT_API] Error details: {str(e)}")
        # Try to use findings directly as last resort
        try:
            assistant_response = _build_fast_fallback_answer(question, findings)
        except Exception as fallback_error:
            print(f"[CHAT_API] Fallback also failed: {fallback_error}")
            assistant_response = "I'm having trouble processing your request. Please try rephrasing your question."

    await create_conversation(session_id, "user", question)
    await create_conversation(session_id, "ai", assistant_response)

    return {"answer": assistant_response}