from __future__ import annotations

import json
import re
from typing import Any, TypedDict, Literal

from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage

# Use the text-only LLM for chat — faster and no image input needed
from app.services.image_analysis_service import safe_text_llm_invoke


class ChatAgentState(TypedDict):
    question: str
    findings: list[dict]
    conversation: list[dict]
    
    relevant_findings: list[dict]
    agent_outputs: dict[str, str]  # { "inspector": "...", "repair": "...", ... }
    next_step: str  # The name of the next agent to call or "FINISH"
    
    action: Literal["clarify", "answer"]
    follow_up_questions: list[str]
    assistant_response: str


_INVALID_ANSWERS = {"not visible in the image", "no answer available"}

# --- Conversational Guardrails ---
GUARDRAILS = """
1. SCOPE: You are a property inspection and repair assistant. ONLY discuss property condition, maintenance, and repair. 
2. SAFETY: For all high-risk hazards (electrical, gas, mold, structural), recommend consulting a licensed professional. 
3. GROUNDING: Use ONLY the findings provided in the context. Do not invent details.
4. TONE: Be helpful, professional, and friendly.
5. LANGUAGE: Use simple, plain English (Grade 6-8 level). Avoid jargon; if you must use a technical term, explain it simply.
6. NO ADVICE: Do not provide legal, financial, or real estate valuation advice.
"""


def _normalize_text(s: str) -> str:
    return (s or "").strip().lower()


def _tokenize(s: str) -> set[str]:
    s = _normalize_text(s)
    tokens = re.split(r"[^a-z0-9]+", s)
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
    q_tokens = _tokenize(question)
    q_text = _normalize_text(item.get("question", ""))
    a_text = _normalize_text(item.get("answer", ""))
    
    # Score based on question matching
    q_score = _tokenize(q_text).intersection(q_tokens)
    
    # Score based on answer content - more lenient
    a_score = _tokenize(a_text).intersection(q_tokens)
    
    # Bonus for specific defect-related terms
    defect_keywords = {"leak", "damage", "crack", "stain", "mold", "rust", "corrosion", "broken", "loose", "missing", "faulty", "defective"}
    question_defects = q_tokens.intersection(defect_keywords)
    answer_defects = _tokenize(a_text).intersection(defect_keywords)
    
    # Calculate total score with bonuses
    base_score = len(q_score) * 2 + len(a_score)
    defect_bonus = len(question_defects) * 3 + len(answer_defects) * 2
    
    return base_score + defect_bonus


def _extract_json_from_text(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1), strict=False)
            except Exception:
                pass
    
    # Heuristic: find the first { and the last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1], strict=False)
        except Exception:
            pass
            
    return json.loads(text, strict=False)


def _infer_response_mode(question: str) -> str:
    q = _normalize_text(question)

    explanation_keywords = {
        "explain", "meaning", "what does this mean",
        "findings", "clarify findings", "interpret",
    }
    summary_keywords = {
        "summary", "summarize", "summarise", "key findings", "what did you find", "what are the findings",
    }
    repair_keywords = {
        "repair", "fix", "steps", "how to", "replace",
        "renovate", "materials", "tools", "cost", "give me", "provide me",
    }
    overall_keywords = {
        "overall", "checklist", "all", "everything",
        "priority", "what should i do", "maintenance plan",
    }

    # Check for repair requests first (highest priority)
    if any(k in q for k in repair_keywords):
        return "repair"
    if any(k in q for k in summary_keywords):
        return "summary"
    if any(k in q for k in overall_keywords):
        return "checklist"
    if any(k in q for k in explanation_keywords):
        return "explain"
    return "explain"


async def select_relevant_findings_node(state: ChatAgentState) -> dict[str, Any]:
    question = state["question"]
    findings = state.get("findings") or []

    print(f"[CHAT_DEBUG] Question: {question}")
    print(f"[CHAT_DEBUG] Total findings received: {len(findings)}")
    for i, f in enumerate(findings[:3]):  # Log first 3 findings
        print(f"[CHAT_DEBUG] Finding {i}: Q='{f.get('question', '')[:50]}...' A='{f.get('answer', '')[:50]}...'")

    valid = _filter_valid_findings(findings)
    print(f"[CHAT_DEBUG] Valid findings after filtering: {len(valid)}")
    
    if not valid:
        # Even if no "valid" findings, try to use partial findings for general guidance
        partial_findings = [f for f in findings if f.get("answer", "").strip() and 
                           f.get("answer", "").strip().lower() not in {"not visible in the image", "no answer available"}]
        print(f"[CHAT_DEBUG] Using partial findings: {len(partial_findings)}")
        return {"relevant_findings": partial_findings[:3]}

    scored = [(it, _score_finding(it, question)) for it in valid]
    scored.sort(key=lambda x: x[1], reverse=True)
    
    # BROAD SEARCH: If it's a summary or overall query, don't filter by keyword score 
    # instead take the most important (defect-heavy) items
    response_mode = _infer_response_mode(question)
    if response_mode in ["summary", "checklist"]:
        print(f"[CHAT_DEBUG] Broad query detected ({response_mode}), including all top valid findings")
        top = [it for it, _ in scored[:50]] # Include up to 50 for broad queries
    else:
        # For specific questions, pick the most relevant
        top = [it for it, score in scored[:15] if score > 0]
        # If no positive matches, fallback to best partial matches
        if not top:
            top = [it for it, _ in scored[:8]]

    print(f"[CHAT_DEBUG] Selected {len(top)} relevant findings for agent context.")
    return {"relevant_findings": top}

# --- Agent Factory ---

async def run_specialist_agent(state: ChatAgentState, persona_name: str, system_prompt: str) -> dict[str, Any]:
    """Helper to run a specialist agent with a specific identity."""
    question = state["question"]
    relevant = state.get("relevant_findings") or []
    
    full_prompt = f"""
    IDENTITY: {system_prompt}
    
    {GUARDRAILS}
    
    USER QUESTION: {question}
    PREVIOUS FINDINGS CONTEXT: {json.dumps(relevant)}
    
    TASK: Provide your expertise on this specific query based ONLY on the findings. 
    Be concise, explain things simply, and stay within your specialty.
    """

    msg = HumanMessage(content=full_prompt)
    response = await safe_text_llm_invoke([msg])
    
    outputs = state.get("agent_outputs") or {}
    outputs[persona_name] = response.content
    
    print(f"[AGENT: {persona_name}] Completed task")
    return {"agent_outputs": outputs}


# --- Specialist Agent Definitions ---

async def inspector_agent(state: ChatAgentState):
    prompt = """You are a Senior Certified Property Inspector. Your job is to analyze the 'What' and 'Why' of inspection findings. Explain issues in simple, easy-to-understand English.
    
    **FORMATTING RULE**: Use Markdown bullet points (-) for each finding or observation. Use clear headers (###).
    """
    return await run_specialist_agent(state, "inspector", prompt)


async def repair_specialist(state: ChatAgentState):
    prompt = """You are a Licensed General Contractor and Master Handyman. Your job is to provide specific, technical, step-by-step repair guidance. Provide:
    1. A step-by-step technical repair plan using a numbered list.
    2. Specific tools and materials typically required using a bulleted list.
    3. Indicators to verify the repair was successful using a bulleted list.
    
    **FORMATTING RULE**: Use Markdown lists (1. for steps, - for items) to make your response easy to scan.
    """
    return await run_specialist_agent(state, "repair", prompt)


async def safety_evaluator(state: ChatAgentState):
    prompt = """You are a Safety and Risk Compliance Auditor. Your job is to identify immediate hazards (electrical, structural, health) and provide critical safety warnings and professional advice. Evaluate for:
    1. Immediate safety hazards (Electrical, Water, Structural).
    2. Potential for mold or long-term damage.
    3. Explicit advice on when to call a specific professional.
    
    **FORMATTING RULE**: Use Markdown bullet points (-) for every risk or safety warning. Use bold text for critical warnings.
    """
    return await run_specialist_agent(state, "safety", prompt)


# --- Master Supervisor / Router ---

async def master_supervisor_node(state: ChatAgentState) -> dict[str, Any]:
    question = state["question"]
    outputs = state.get("agent_outputs") or {}
    
    prompt = f"""
    You are the Master Supervisor of a Property Inspection Team.
    Your goal is to coordinate specialists to answer the user's question completely.
    
    USER QUESTION: {question}
    SPECIALISTS AVAILABLE: [inspector, repair, safety]
    TEAM PROGRESS SO FAR: {list(outputs.keys())}
    
    DECISION RULES:
    1. If the question mentions "how to fix", "repair", or "steps" and 'repair' hasn't spoken -> call 'repair'.
    2. If the prompt is "summarize", "explain", or "what's wrong" and 'inspector' hasn't spoken -> call 'inspector'.
    3. If there are signs of leaks, mold, electrical, or structural risk and 'safety' hasn't spoken -> call 'safety'.
    4. If the team has gathered all necessary info to answer the user -> go to 'synthesizer'.
    
    Return ONLY valid JSON:
    {{
        "next_step": "inspector" | "repair" | "safety" | "synthesizer"
    }}
    """
    
    try:
        msg = HumanMessage(content=prompt)
        response = await safe_text_llm_invoke([msg])
        parsed = _extract_json_from_text(response.content)
        next_step = parsed.get("next_step", "synthesizer")
    except Exception:
        next_step = "synthesizer"
        
    print(f"[SUPERVISOR] Next specialist chosen: {next_step}")
    return {"next_step": next_step}


async def synthesizer_node(state: ChatAgentState) -> dict[str, Any]:
    question = state["question"]
    outputs = state.get("agent_outputs") or {}
    
    prompt = f"""
    You are the Senior Property Synthesis Agent. Your job is to curate a single, high-quality response from the team's specialist reports.
    
    {GUARDRAILS}
    
    Analyze these findings and explain:
    1. What the issue is in simple English.
    2. How it affects the home.
    3. What other systems might be involved.
    
    **FORMATTING RULE**: Use Markdown bullet points (-) for each finding. Use clear headers (###).
    
    USER QUESTION: {question}
    REPORTS FROM SPECIALISTS: {json.dumps(outputs)}
    
    RULES:
    1. Use simple words and short sentences.
    2. Organize with logical Markdown headers (###).
    3. Combine overlapping info into clear sections.
    4. **CRITICAL**: Use Markdown bullet points (-) for findings and instructions. Avoid long paragraphs.
    5. Always ensure Safety advice is prominent and clear.
    6. End with a friendly, simple follow-up question.
    
    Output MUST be in valid JSON:
    {{"assistant_response": "merged markdown content", "action": "answer", "follow_up_questions": ["..."]}}
    """
    
    try:
        msg = HumanMessage(content=prompt)
        response = await safe_text_llm_invoke([msg])
        parsed = _extract_json_from_text(response.content)
    except Exception:
        # Fallback to crude merge if synthesizer fails
        merged = "\n\n".join(outputs.values())
        return {"assistant_response": merged, "action": "answer", "follow_up_questions": []}
    
    return {
        "assistant_response": parsed.get("assistant_response", ""),
        "action": parsed.get("action", "answer"),
        "follow_up_questions": parsed.get("follow_up_questions", [])
    }


def build_chat_agent_graph():
    workflow = StateGraph(ChatAgentState)
    
    # ── Nodes ──
    workflow.add_node("select_relevant_findings", select_relevant_findings_node)
    workflow.add_node("supervisor", master_supervisor_node)
    workflow.add_node("inspector", inspector_agent)
    workflow.add_node("repair", repair_specialist)
    workflow.add_node("safety", safety_evaluator)
    workflow.add_node("synthesizer", synthesizer_node)

    # ── Non-Sequential Hub-and-Spoke Logic ──
    workflow.set_entry_point("select_relevant_findings")
    workflow.add_edge("select_relevant_findings", "supervisor")
    
    # Dynamic routing from supervisor
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["next_step"],
        {
            "inspector": "inspector",
            "repair": "repair",
            "safety": "safety",
            "synthesizer": "synthesizer"
        }
    )
    
    # After a worker finishes, always go back to supervisor for next orders
    workflow.add_edge("inspector", "supervisor")
    workflow.add_edge("repair", "supervisor")
    workflow.add_edge("safety", "supervisor")
    
    workflow.set_finish_point("synthesizer")
    return workflow.compile()


chat_graph = build_chat_agent_graph()