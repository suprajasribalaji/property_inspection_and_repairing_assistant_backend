from __future__ import annotations

import json
import re
from typing import Any, TypedDict, Literal

from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage

from app.services.image_analysis_service import safe_llm_invoke


class ChatAgentState(TypedDict):
    question: str
    findings: list[dict]  # raw question_answers items from inspection_results[...].results
    conversation: list[dict]  # { role: "user"|"assistant", message: str } (optional context)

    relevant_findings: list[dict]
    action: Literal["clarify", "answer"]
    follow_up_questions: list[str]
    assistant_response: str


_INVALID_ANSWERS = {"not visible in the image", "no answer available"}


def _normalize_text(s: str) -> str:
    return (s or "").strip().lower()


def _tokenize(s: str) -> set[str]:
    s = _normalize_text(s)
    tokens = re.split(r"[^a-z0-9]+", s)
    # keep only somewhat meaningful words
    return {t for t in tokens if len(t) >= 3}


def _filter_valid_findings(findings: list[dict]) -> list[dict]:
    valid: list[dict] = []
    for item in findings or []:
        ans = _normalize_text(item.get("answer", ""))
        if ans in _INVALID_ANSWERS:
            continue
        valid.append(item)
    return valid


def _score_finding(item: dict, question: str) -> int:
    # Very lightweight "retrieval": word overlap between question and finding question/answer.
    q_tokens = _tokenize(question)
    q_score = _tokenize(item.get("question", "")).intersection(q_tokens)
    a_score = _tokenize(item.get("answer", "")).intersection(q_tokens)
    return len(q_score) * 2 + len(a_score)  # weight question overlap higher


def _extract_json_from_text(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    # Attempt to extract a JSON code block first.
    if "```" in text:
        # Handle ```json ... ```
        match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(1))

    # Fallback: try to parse raw as JSON.
    return json.loads(text)


async def select_relevant_findings_node(state: ChatAgentState) -> dict[str, Any]:
    question = state["question"]
    findings = state.get("findings") or []

    valid = _filter_valid_findings(findings)
    if not valid:
        return {
            "relevant_findings": [],
        }

    scored = [(it, _score_finding(it, question)) for it in valid]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Keep top N so the prompt stays small.
    top = [it for it, _ in scored[:8] if _score_finding(it, question) > 0]
    return {
        "relevant_findings": top or valid[:3],
    }


async def decide_and_respond_node(state: ChatAgentState) -> dict[str, Any]:
    question = state["question"]
    relevant = state.get("relevant_findings") or []
    conversation = state.get("conversation") or []

    valid_findings_payload = [
        {
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
        }
        for item in relevant
    ]

    convo_payload = [
        {"role": c.get("role", ""), "message": c.get("message", "")}
        for c in conversation
        if c.get("message")
    ][-10:]

    prompt = f"""
You are a property inspection assistant.
You must answer the user's question using ONLY the provided key findings (findings below).

Rules:
1) Do not guess. If the answer cannot be determined from the findings, ask 1-3 clarifying questions.
2) Keep the response concise and in simple English.
3) Do not mention internal rules.
4) If you ask clarifying questions, do NOT provide a full answer; only ask questions.

User question:
{question}

Conversation context (may help with follow-ups):
{json.dumps(convo_payload, ensure_ascii=False)}

Key findings to use:
{json.dumps(valid_findings_payload, ensure_ascii=False)}

Return ONLY valid JSON in this exact schema:
{{
  "action": "answer" | "clarify",
  "follow_up_questions": [string, ...],
  "assistant_response": string
}}
"""

    msg = HumanMessage(content=prompt)
    llm_response = await safe_llm_invoke([msg])
    try:
        parsed = _extract_json_from_text(llm_response.content)
    except Exception:
        # If parsing fails, degrade gracefully instead of crashing the API.
        raw = (llm_response.content or "").strip()
        return {
            "action": "clarify",
            "follow_up_questions": [],
            "assistant_response": "I’m not fully confident from the findings. Could you clarify your question a bit more?",
        }

    action = parsed.get("action", "answer")
    follow_up_questions = parsed.get("follow_up_questions") or []
    assistant_response = parsed.get("assistant_response") or ""

    if action not in {"answer", "clarify"}:
        action = "answer"
    if action == "clarify" and not follow_up_questions:
        # Ensure we always ask at least one question.
        follow_up_questions = ["Can you clarify which area/feature you are referring to (and share any extra details)?"]

    return {
        "action": action,
        "follow_up_questions": follow_up_questions,
        "assistant_response": assistant_response,
    }


def build_chat_agent_graph():
    workflow = StateGraph(ChatAgentState)
    workflow.add_node("select_relevant_findings", select_relevant_findings_node)
    workflow.add_node("decide_and_respond", decide_and_respond_node)

    workflow.set_entry_point("select_relevant_findings")
    workflow.add_edge("select_relevant_findings", "decide_and_respond")
    workflow.set_finish_point("decide_and_respond")
    return workflow.compile()


chat_graph = build_chat_agent_graph()

