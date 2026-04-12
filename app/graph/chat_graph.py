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


def _infer_response_mode(question: str) -> str:
    q = _normalize_text(question)

    explanation_keywords = {
        "explain",
        "meaning",
        "what does this mean",
        "summary",
        "findings",
        "clarify findings",
        "interpret",
    }
    repair_keywords = {
        "repair",
        "fix",
        "steps",
        "how to",
        "replace",
        "renovate",
        "materials",
        "tools",
        "cost",
    }
    overall_keywords = {
        "overall",
        "checklist",
        "all",
        "everything",
        "priority",
        "what should i do",
        "maintenance plan",
    }

    if any(k in q for k in overall_keywords):
        return "checklist"
    if any(k in q for k in repair_keywords):
        return "repair"
    if any(k in q for k in explanation_keywords):
        return "explain"
    return "explain"


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
    response_mode = _infer_response_mode(question)

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
You are a practical Property Inspection & Repair Assistant.

You are helping a homeowner based on an inspection checklist and the resulting "key findings" (question/answer pairs).
You should be actionable and practical, using general home-repair best practices, while staying consistent with the provided findings.

Inspection domains (the checklist used to generate findings):
I. General Structure & Safety
II. Plumbing & Water
III. Electrical Systems
IV. Kitchen Inspection
V. Bathrooms
VI. Bedrooms
VII. Living Room & Common Areas
VIII. HVAC & Utilities
IX. Basement, Attic & Exterior

Core behavior:
1) First identify user intent and adapt output:
   - mode=explain: explain findings in plain language only; do NOT include repair plan/tools unless user asks.
   - mode=repair: provide practical repair steps for the asked area.
   - mode=checklist: provide a prioritized cross-domain checklist.
2) Current detected mode: {response_mode}
3) If the mode is checklist (or user asks broadly), provide a prioritized checklist across domains, and clearly mark:
   - "Supported by findings" vs
   - "Inspect/confirm" (not proven by findings)
4) Ask 1–3 clarifying questions only when missing info changes safety/cost/approach (active leak, electrical hazard, suspected mold, structural movement).

Response style (be consistent):
1) For mode=explain:
   - Only provide "What this likely means": 2–5 bullets
   - Do NOT provide repair steps, tools, checklist, or monitoring section
   - End with exactly one follow-up question asking what the user wants next
     (example: "Would you like repair steps for any specific issue?")

2) For mode=repair:
   - Do NOT repeat or re-summarize the findings — the user already knows what the problem is. Go straight to the fix.
   - Do NOT include a "What this likely means" section.
   - "Practical step-by-step plan": numbered steps
   - "Tools / materials (typical)": short list (if relevant)
   - "How to verify it's fixed": 1–3 checks
   - "Safety / when to call a pro": 1–3 bullets

3) For mode=checklist:
   - "Priority actions (now / soon / later)"
   - Include supported vs inspect/confirm labeling

Safety gates (do not ignore):
- Electrical (warm outlets/switches, sparking, buzzing, burning smell, repeated breaker trips): recommend a licensed electrician.
- Plumbing/moisture (water stains, damp walls, leaks): stop the water source and dry fully before patch/paint.
- Mold (musty smell, visible growth, large affected area): recommend proper remediation and PPE.
- Structure (foundation cracks, sloping floors, sagging ceiling): recommend professional evaluation if severe/worsening.

Grounding:
- Use the key findings below as your starting point.
- It is OK to provide general best-practice steps even if some details are not in findings, but do not invent facts about this specific home.
- When a step depends on unknown details, say so briefly (e.g., "If this is drywall..." / "If the stain is from an active leak...").

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
        return {
            "action": "answer",
            "follow_up_questions": [],
            "assistant_response": (
                "Here are some general, practical steps you can follow based on typical home inspection and repair "
                "best practices. If you share more details (for example interior vs exterior, materials, and whether "
                "moisture is already fixed), I can refine these steps further."
            ),
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

