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

class RateLimitTimeoutError(Exception):
    """Custom exception for when we wait too long for rate limits."""
    pass

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Default to common Groq models if env vars are missing
GROQ_VISION_MODEL_NAME = os.getenv("GROQ_VISION_MODEL_NAME", "llama-3.2-11b-vision-preview")
GROQ_TEXT_MODEL_NAME = os.getenv("GROQ_TEXT_MODEL_NAME", "llama-3.3-70b-versatile")

MAX_REQUESTS_PER_DAY = int(os.getenv("MAX_REQUESTS_PER_DAY", 1000))
MIN_DELAY_BETWEEN_CALLS = float(os.getenv("MIN_DELAY_BETWEEN_CALLS", 0.5))

# Vision-capable model — used for all image analysis tasks
vision_llm = ChatGroq(
    model=GROQ_VISION_MODEL_NAME,
    api_key=GROQ_API_KEY,
    temperature=0.1,
)

# Fast text-only model — used for the chat assistant (no image input)
text_llm = ChatGroq(
    model=GROQ_TEXT_MODEL_NAME,
    api_key=GROQ_API_KEY,
    temperature=0.1,
)

# Rate limiting variables
last_api_call_time = 0
api_calls_today = 0
api_calls_reset_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

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
    
    # Wait for rate limit / usage capacity
    max_wait = 60
    waited = 0
    while True:
        can_make, reason, wait_seconds = usage_tracker.can_make_request()
        if can_make:
            break
        
        if "Daily limit" in reason:
            raise Exception(f"API usage limit reached: {reason}")
            
        print(f"[LLM_INVOKE] Rate limit hit, waiting {wait_seconds:.1f}s. Reason: {reason}")
        sleep_chunk = min(wait_seconds, 10) # sleep in chunks
        await asyncio.sleep(sleep_chunk)
        waited += sleep_chunk
        if waited > max_wait:
            raise RateLimitTimeoutError(f"Many people are using the system right now. Please try again after some time (about 1 minute) to allow the services to reset.")

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

    # Wait for rate limit / usage capacity
    max_wait = 60
    waited = 0
    while True:
        can_make, reason, wait_seconds = usage_tracker.can_make_request()
        if can_make:
            break
            
        if "Daily limit" in reason:
            raise Exception(f"API usage limit reached: {reason}")

        print(f"[TEXT_LLM_INVOKE] Rate limit hit, waiting {wait_seconds:.1f}s. Reason: {reason}")
        sleep_chunk = min(wait_seconds, 10)
        await asyncio.sleep(sleep_chunk)
        waited += sleep_chunk
        if waited > max_wait:
            raise Exception(f"Timed out waiting for rate limit: {reason}")

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

    prompt = """You are a highly detailed and professional property inspection AI. 
EXTREMELY IMPORTANT: Your goal is to identify every possible detail in this image to support 100+ standard inspection questions. Do not be brief. Be pixel-perfect and exhaustive.

Carefully examine every visible detail and produce a comprehensive list of observations across these categories:

1. STRUCTURAL & SURFACES:
   - Walls, ceilings, floors: Note any cracks (even hairline), stains (water, mold), peeling paint, buckling, or unevenness.
   - Stairs & Railings: Check for security, gaps, and condition.

2. OPENINGS & EXTERIOR:
   - Windows & Doors: Note condition of frames, seals, glass (cracks/fogging), locks, and screens.
   - Exterior: Siding, trim, masonry, gutters, downspouts, and signs of drainage issues.
   - Roof (if visible): Shingle condition, flashing, and debris.

3. SYSTEMS (ELECTRICAL, PLUMBING, HVAC):
   - Electrical: Outlets, switches, panels, visible wiring, and lighting fixtures. Note if they look modern or outdated.
   - Plumbing: Faucets, drains, supply lines, moisture under sinks, water heater condition, and rust.
   - HVAC: Vents, HVAC units, filters (if visible), and thermostat location/condition.

4. KITCHEN & BATHROOM:
   - Cabinets, countertops, backsplash, and appliances (brand, condition, visible damage).
   - Tiles, grout condition, and signs of moisture or poor ventilation.

5. SAFETY & GENERAL:
   - Hazards: Tripping hazards, exposed wires, sharp edges, or fire risks (smoke detectors).
   - Cleanliness and maintenance level.

RULES:
- Be hyper-specific. Instead of "stain on wall", say "3-inch circular yellowish-brown water stain on upper right corner of the north wall".
- Note materials where possible (e.g., "hardwood flooring", "granite countertops", "copper piping").
- Mention if something is in EXCELLENT condition as well as defects.
- DO NOT speculate about what you cannot see, but report EVERYTHING you can see.

Return ONLY valid JSON with no markdown or extra text:
{
    "observations": [
        "precise observation 1",
        "precise observation 2",
        ...
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

    prompt = f"""You are an expert AI property inspector. Use the observations and the image to answer these questions with high precision.

PRIOR OBSERVATIONS (from this same image):
{obs_text}

CRITICAL RULES:
1. ACCURACY: Answer based ONLY on the image and observations. 
2. EXPLANATION: Every answer MUST include at least one sentence of detailed explanation describing what is visible. DO NOT just say "Yes" or "No".
3. NO HALLUCINATION: If a feature is not visible, say "Not visible in the image".
4. SAFETY GUARDRAIL: For dangerous issues (exposed wiring, major structural cracks, gas leaks), ALWAYS append a warning: "Recommended action: Consult a licensed professional immediately."
5. NO LEGAL/FINANCIAL ADVICE: Do not estimate property value or provide legal opinions on compliance. Focus purely on physical condition.
6. STYLE: Use plain, simple English that anyone can understand.
7. VOLUME: You MUST return exactly {n} answers, one for each question.

Questions:
{question_text}

Return ONLY valid JSON with no markdown or code fences:
{{
  "answers": [
    "simple, accurate answer 1",
    "simple, accurate answer 2"
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