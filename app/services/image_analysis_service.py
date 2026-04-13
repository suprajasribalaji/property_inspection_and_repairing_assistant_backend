from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import os
import base64
import json
import asyncio
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from app.services.api_usage_tracker import usage_tracker

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Vision-capable model — used for all image analysis tasks
vision_llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=GROQ_API_KEY,
    temperature=0.1,
)

# Fast text-only model — used for the chat assistant (no image input)
text_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0.1,
)

# Rate limiting variables
last_api_call_time = 0
api_calls_today = 0
api_calls_reset_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
MAX_REQUESTS_PER_DAY = 1000   # Groq free tier is much more generous than Gemini
MIN_DELAY_BETWEEN_CALLS = 0.5  # slightly conservative to avoid burst 429s

# Chunk size for answering questions — ≤20 per call keeps token count manageable
QUESTION_CHUNK_SIZE = 20


def build_image_message(prompt: str, image_bytes: bytes, mime_type: str) -> HumanMessage:
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    image_url = f"data:{mime_type};base64,{b64_image}"

    return HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": image_url},
            },
        ]
    )


async def check_rate_limit():
    print("[RATE_LIMIT] Starting rate limit check")
    global last_api_call_time, api_calls_today, api_calls_reset_time

    now = datetime.now()
    print(f"[RATE_LIMIT] Current time: {now}, API calls today: {api_calls_today}/{MAX_REQUESTS_PER_DAY}")

    if now >= api_calls_reset_time:
        print(f"[RATE_LIMIT] Resetting daily counter, was: {api_calls_today}")
        api_calls_today = 0
        api_calls_reset_time = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        print(f"[RATE_LIMIT] New reset time: {api_calls_reset_time}")

    if api_calls_today >= MAX_REQUESTS_PER_DAY:
        wait_time = (api_calls_reset_time - now).total_seconds()
        print(f"[RATE_LIMIT] Daily quota exceeded, wait time: {wait_time:.0f}s")
        raise Exception(f"Daily API quota exceeded. Please wait {wait_time:.0f} seconds or upgrade your plan.")

    time_since_last_call = now.timestamp() - last_api_call_time
    print(f"[RATE_LIMIT] Time since last call: {time_since_last_call:.2f}s, min delay: {MIN_DELAY_BETWEEN_CALLS}s")
    if time_since_last_call < MIN_DELAY_BETWEEN_CALLS:
        sleep_time = MIN_DELAY_BETWEEN_CALLS - time_since_last_call
        print(f"[RATE_LIMIT] Sleeping for {sleep_time:.2f}s")
        await asyncio.sleep(sleep_time)

    last_api_call_time = datetime.now().timestamp()
    api_calls_today += 1
    print(f"[RATE_LIMIT] API call #{api_calls_today} approved, last call time updated")


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "rate limit" in msg or "429" in msg or "too many requests" in msg


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(Exception),
)
async def safe_llm_invoke(messages):
    """Invoke the vision LLM (used for image-based tasks)."""
    print("[LLM_INVOKE] Starting safe LLM invoke (vision model)")
    can_make, reason = usage_tracker.can_make_request()
    print(f"[LLM_INVOKE] Usage tracker check: can_make={can_make}, reason={reason}")
    if not can_make:
        raise Exception(f"API usage limit reached: {reason}")

    print("[LLM_INVOKE] Checking rate limits")
    await check_rate_limit()

    print("[LLM_INVOKE] Invoking vision LLM API")
    start_time = time.time()
    try:
        response = await vision_llm.ainvoke(messages)
        response_time = time.time() - start_time
        print(f"[LLM_INVOKE] LLM call successful in {response_time:.2f}s")
        usage_tracker.log_api_call("groq_vision_api", True, response_time)
        return response
    except Exception as e:
        response_time = time.time() - start_time
        print(f"[LLM_INVOKE] Exception after {response_time:.2f}s: {e}")
        usage_tracker.log_api_call("groq_vision_api", False, response_time)

        if "RESOURCE_EXHAUSTED" in str(e) or _is_rate_limit_error(e) or "limit reached" in str(e).lower():
            print("[LLM_INVOKE] Quota / rate-limit exceeded")
            raise Exception("API quota exceeded. Please try again later or upgrade to a paid plan.")
        raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(Exception),
)
async def safe_text_llm_invoke(messages):
    """Invoke the text-only LLM (used by the chat assistant — no image content)."""
    print("[TEXT_LLM_INVOKE] Starting safe text LLM invoke")
    can_make, reason = usage_tracker.can_make_request()
    print(f"[TEXT_LLM_INVOKE] Usage tracker check: can_make={can_make}, reason={reason}")
    if not can_make:
        raise Exception(f"API usage limit reached: {reason}")

    print("[TEXT_LLM_INVOKE] Checking rate limits")
    await check_rate_limit()

    print("[TEXT_LLM_INVOKE] Invoking text LLM API")
    start_time = time.time()
    try:
        response = await text_llm.ainvoke(messages)
        response_time = time.time() - start_time
        print(f"[TEXT_LLM_INVOKE] LLM call successful in {response_time:.2f}s")
        usage_tracker.log_api_call("groq_text_api", True, response_time)
        return response
    except Exception as e:
        response_time = time.time() - start_time
        print(f"[TEXT_LLM_INVOKE] Exception after {response_time:.2f}s: {e}")
        usage_tracker.log_api_call("groq_text_api", False, response_time)

        if "RESOURCE_EXHAUSTED" in str(e) or _is_rate_limit_error(e) or "limit reached" in str(e).lower():
            print("[TEXT_LLM_INVOKE] Quota / rate-limit exceeded")
            raise Exception("API quota exceeded. Please try again later or upgrade to a paid plan.")
        raise


# ---------------------------------------------------------------------------
# Observation extraction — rich grounding pass before answering questions
# ---------------------------------------------------------------------------

async def extract_observations(image_bytes: bytes, mime_type: str) -> list[str]:
    print(f"[EXTRACT_OBS] Starting, mime_type: {mime_type}, image_size: {len(image_bytes)} bytes")

    prompt = """You are a professional property inspection AI assistant.

Carefully examine every visible detail in this property image and produce a comprehensive list of observations.

Focus on ALL of the following areas if visible:
- Walls, ceilings, and floors: cracks, stains, water damage, peeling paint, mold, discoloration
- Doors and windows: condition, gaps, damage, locks, seals
- Electrical: outlets, switches, panels, wiring, fixtures
- Plumbing: pipes, fixtures, faucets, drains, water stains
- Kitchen: appliances, cabinets, countertops, sink, ventilation
- Bathroom: toilet, shower, tub, tiles, grout, ventilation
- Structural elements: beams, columns, stairs, railings
- HVAC: vents, units, ducts, filters
- General cleanliness, maintenance level, and safety hazards

Rules:
- Describe EXACTLY what you see — be specific and detailed
- Note the condition of each item (good, fair, poor, damaged, etc.)
- Include severity where estimable (e.g. "hairline crack", "large water stain ~30 cm diameter")
- Do not omit anything visible, even minor items
- Do NOT guess about hidden issues

Return ONLY valid JSON with no markdown or extra text:
{
    "observations": [
        "observation 1",
        "observation 2"
    ]
}"""

    message = build_image_message(prompt, image_bytes, mime_type)
    try:
        print("[EXTRACT_OBS] Calling LLM")
        response = await safe_llm_invoke([message])
        raw = response.content.strip()
        if "```" in raw:
            raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        observations = data.get("observations", [])
        print(f"[EXTRACT_OBS] Extracted {len(observations)} observations")
        return observations
    except Exception as e:
        print(f"[EXTRACT_OBS] Error: {e}")
        if "quota exceeded" in str(e).lower() or "limit reached" in str(e).lower():
            raise
        return []


# ---------------------------------------------------------------------------
# Chunked question answering — every question gets a real answer
# ---------------------------------------------------------------------------

async def answer_question_chunk(
    image_bytes: bytes,
    mime_type: str,
    chunk_questions: list[str],
    chunk_start_idx: int,
    observations: list[str],
) -> list[str]:
    """Answer a small chunk of questions against the image in a single LLM call."""
    obs_text = "\n".join(f"- {o}" for o in observations) if observations else "No prior observations available."
    question_text = "\n".join(
        [f"{chunk_start_idx + i + 1}. {q}" for i, q in enumerate(chunk_questions)]
    )
    n = len(chunk_questions)

    prompt = f"""You are an expert AI property inspector. Answer every question about this property image as accurately and specifically as possible.

PRIOR OBSERVATIONS (from this same image):
{obs_text}

CRITICAL RULES:
1. Examine the image carefully AND use the observations above together.
2. You MUST return exactly {n} answers — one for each numbered question below.
3. Answering style:
   - Visible feature in GOOD condition → describe it specifically (colour, material, location, size).
   - Visible feature with ISSUES → describe the defect specifically (type, severity, location).
   - PARTIALLY visible or UNCERTAIN → give your best assessment and note the uncertainty.
   - GENUINELY NOT VISIBLE → write "Not visible in the image" (use sparingly).
4. Each answer must be 1–4 concise sentences in plain English.
5. Do NOT use vague filler like "appears to be fine" without supporting detail.

Questions:
{question_text}

Return ONLY valid JSON with no markdown or code fences:
{{
  "answers": [
    "answer 1",
    "answer 2"
  ]
}}

The "answers" array must have exactly {n} elements in the same order as the questions."""

    message = build_image_message(prompt, image_bytes, mime_type)
    try:
        response = await safe_llm_invoke([message])
        raw = response.content.strip()
        if "```" in raw:
            raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        answers = data.get("answers", [])
        # Pad or trim to exact chunk size
        while len(answers) < n:
            answers.append("Not visible in the image")
        return [str(a).strip() if a else "Not visible in the image" for a in answers[:n]]
    except Exception as e:
        print(f"[CHUNK_ANSWER] Error for chunk starting at {chunk_start_idx}: {e}")
        if "quota exceeded" in str(e).lower() or "limit reached" in str(e).lower():
            raise
        # Keyword-based fallback
        obs_joined = " ".join(observations).lower()
        fallback = []
        for q in chunk_questions:
            keywords = [w for w in q.lower().split() if len(w) > 3]
            if any(kw in obs_joined for kw in keywords):
                fallback.append("Based on visible observations, this area requires further inspection.")
            else:
                fallback.append("Not visible in the image")
        return fallback


async def answer_all_questions_chunked(
    image_bytes: bytes,
    mime_type: str,
    questions: list[str],
    observations: list[str],
) -> list[str]:
    """
    Answer ALL questions by splitting into chunks of QUESTION_CHUNK_SIZE.
    Processed sequentially to respect Groq rate limits.
    """
    total = len(questions)
    print(f"[CHUNKED] Answering {total} questions in chunks of {QUESTION_CHUNK_SIZE}")

    full_answers = ["Not visible in the image"] * total
    chunks = [
        (i, questions[i: i + QUESTION_CHUNK_SIZE])
        for i in range(0, total, QUESTION_CHUNK_SIZE)
    ]
    print(f"[CHUNKED] Created {len(chunks)} chunks")

    for chunk_start, chunk_qs in chunks:
        print(f"[CHUNKED] Chunk start={chunk_start}, size={len(chunk_qs)}")
        try:
            chunk_answers = await answer_question_chunk(
                image_bytes, mime_type, chunk_qs, chunk_start, observations
            )
            for j, ans in enumerate(chunk_answers):
                full_answers[chunk_start + j] = ans
            print(f"[CHUNKED] Chunk at {chunk_start} complete")
        except Exception as e:
            print(f"[CHUNKED] Chunk at {chunk_start} failed: {e}")
            if "quota exceeded" in str(e).lower() or "limit reached" in str(e).lower():
                raise
            # Leave "Not visible" defaults for failed chunk and continue

    valid = sum(
        1 for a in full_answers
        if a.lower() not in {"not visible in the image", "no answer available"}
    )
    print(f"[CHUNKED] Done: {valid}/{total} questions answered with real content")
    return full_answers


# ---------------------------------------------------------------------------
# Public API used by the LangGraph workflow
# ---------------------------------------------------------------------------

async def observe_property_image(image_bytes: bytes, mime_type: str) -> list[str]:
    print(f"[OBSERVE] size={len(image_bytes)} bytes, mime={mime_type}")
    observations = await extract_observations(image_bytes, mime_type)
    print(f"[OBSERVE] Got {len(observations)} observations")
    return observations


async def analyze_property_image(
    observations: list[str],
    image_bytes: bytes,
    mime_type: str,
    questions: list[str],
) -> dict:
    print(f"[ANALYZE] {len(observations)} observations, {len(questions)} questions")

    full_answers = await answer_all_questions_chunked(
        image_bytes, mime_type, questions, observations
    )

    answers_json = json.dumps({"answers": full_answers})
    print("[ANALYZE] Done")
    return {
        "observations": observations,
        "answers": answers_json,
    }