from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
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

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.1
)

# Rate limiting variables
last_api_call_time = 0
api_calls_today = 0
api_calls_reset_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
MAX_REQUESTS_PER_DAY = 20  # Stay under the 20 limit
MIN_DELAY_BETWEEN_CALLS = 0.5  # Seconds between calls


def build_image_message(prompt, image_bytes, mime_type):

    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    image_url = f"data:{mime_type};base64,{b64_image}"

    return HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                },
            },
        ]
    )


async def check_rate_limit():
    print("[RATE_LIMIT] Starting rate limit check")
    global last_api_call_time, api_calls_today, api_calls_reset_time
    
    now = datetime.now()
    print(f"[RATE_LIMIT] Current time: {now}, API calls today: {api_calls_today}/{MAX_REQUESTS_PER_DAY}")
    
    # Reset counter if we've passed the reset time
    if now >= api_calls_reset_time:
        print(f"[RATE_LIMIT] Resetting daily counter, was: {api_calls_today}")
        api_calls_today = 0
        api_calls_reset_time = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        print(f"[RATE_LIMIT] New reset time: {api_calls_reset_time}")
    
    # Check if we've exceeded daily limit
    if api_calls_today >= MAX_REQUESTS_PER_DAY:
        wait_time = (api_calls_reset_time - now).total_seconds()
        print(f"[RATE_LIMIT] Daily quota exceeded, wait time: {wait_time:.0f}s")
        raise Exception(f"Daily API quota exceeded. Please wait {wait_time:.0f} seconds or upgrade your plan.")
    
    # Add delay between calls
    time_since_last_call = now.timestamp() - last_api_call_time
    print(f"[RATE_LIMIT] Time since last call: {time_since_last_call:.2f}s, min delay: {MIN_DELAY_BETWEEN_CALLS}s")
    if time_since_last_call < MIN_DELAY_BETWEEN_CALLS:
        sleep_time = MIN_DELAY_BETWEEN_CALLS - time_since_last_call
        print(f"[RATE_LIMIT] Sleeping for {sleep_time:.2f}s")
        await asyncio.sleep(sleep_time)
    
    last_api_call_time = datetime.now().timestamp()
    api_calls_today += 1
    print(f"[RATE_LIMIT] API call #{api_calls_today} approved, last call time updated")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(ChatGoogleGenerativeAIError)
)
async def safe_llm_invoke(messages):
    print("[LLM_INVOKE] Starting safe LLM invoke")
    # Check usage tracker first
    can_make, reason = usage_tracker.can_make_request()
    print(f"[LLM_INVOKE] Usage tracker check: can_make={can_make}, reason={reason}")
    if not can_make:
        raise Exception(f"API usage limit reached: {reason}")
    
    print("[LLM_INVOKE] Checking rate limits")
    await check_rate_limit()
    
    print("[LLM_INVOKE] Invoking LLM API")
    start_time = time.time()
    try:
        response = await llm.ainvoke(messages)
        response_time = time.time() - start_time
        print(f"[LLM_INVOKE] LLM call successful in {response_time:.2f}s")
        usage_tracker.log_api_call("gemini_api", True, response_time)
        return response
    except ChatGoogleGenerativeAIError as e:
        response_time = time.time() - start_time
        print(f"[LLM_INVOKE] ChatGoogleGenerativeAIError after {response_time:.2f}s: {e}")
        usage_tracker.log_api_call("gemini_api", False, response_time)
        
        if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
            # Handle quota exceeded specifically
            print("[LLM_INVOKE] Resource exhausted/quota exceeded")
            raise Exception("API quota exceeded. Please try again later or upgrade to a paid plan.")
        else:
            # Re-raise for retry logic
            print("[LLM_INVOKE] Retrying due to ChatGoogleGenerativeAIError")
            raise
    except Exception as e:
        response_time = time.time() - start_time
        print(f"[LLM_INVOKE] General exception after {response_time:.2f}s: {e}")
        usage_tracker.log_api_call("gemini_api", False, response_time)
        raise


async def extract_observations(image_bytes: bytes, mime_type: str):
    print(f"[EXTRACT_OBS] Starting observation extraction, mime_type: {mime_type}, image_size: {len(image_bytes)} bytes")

    prompt = """
        You are a property observation and extraction AI assistant.

        Look carefully at the property image and list only the visible observations from the image.

        Rules:
        - Only describe what is clearly visible in the image
        - Do not guess
        - Do not assume anything not visible

        Return ONLY valid JSON in this format:

        {
            "observations": [
                "observation 1",
                "observation 2"
            ]
        }
    """

    print("[EXTRACT_OBS] Building image message")
    message = build_image_message(prompt, image_bytes, mime_type)

    try:
        print("[EXTRACT_OBS] Calling LLM for observations")
        response = await safe_llm_invoke([message])
        print(f"[EXTRACT_OBS] Got response, parsing JSON: {response.content[:100]}...")
        data = json.loads(response.content)
        observations = data["observations"]
        print(f"[EXTRACT_OBS] Successfully extracted {len(observations)} observations")
        return observations
    except Exception as e:
        print(f"[EXTRACT_OBS] Error: {e}")
        if "quota exceeded" in str(e).lower() or "limit reached" in str(e).lower():
            print("[EXTRACT_OBS] Re-raising quota error")
            raise
        print("[EXTRACT_OBS] Returning empty observations due to error")
        return []


async def answer_questions(image_bytes: bytes, mime_type: str, questions, observations):
    print(f"[ANSWER_Q] Starting to answer {len(questions)} questions with {len(observations)} observations")

    question_text = "\n".join(
        [f"{i+1}. {q}" for i, q in enumerate(questions)]
    )

    obs_text = "\n".join(observations)
    print(f"[ANSWER_Q] Question text length: {len(question_text)}, Observations text length: {len(obs_text)}")

    prompt = f"""

        You are an AI property inspection assistant.

        Your task is to analyze a property image and answer inspection questions using ONLY the provided observations and visible information in the image.
        
        STRICT RULES:

            1. Only use information that is explicitly present in:
                - the provided observations
                - the visible content of the image

            2. Do NOT:
                - guess
                - assume
                - infer beyond what is visible
                - use external knowledge

            3.  If the answer cannot be fully determined:
                - Answer using ONLY the available observations related to the question
                - If no relevant observation exists, respond exactly with:
                "Not visible in the image"

            4. Keep the answers simple, short and use simple english language

            5. Keep answers:
            - short but detailed enough to be useful
            - clear
            - in simple English
            - free of technical jargon

            6. Be consistent:
            - Do not change answer style between questions
            - Do not include explanations unless clearly visible

            7. Return ONLY valid JSON in this format:
            - Do not include any additional text or markdown formatting or code blocks.

            {{
                "answers": [
                    "answer to question 1",
                    "answer to question 2",
                    "answer to question 3"
                ]
            }}

            Observations from the image:
            {obs_text}

            Questions:
            {question_text}
    """

    print("[ANSWER_Q] Building image message for questions")
    message = build_image_message(prompt, image_bytes, mime_type)

    try:
        print("[ANSWER_Q] Calling LLM to answer questions")
        response = await safe_llm_invoke([message])
        print(f"[ANSWER_Q] Got response from LLM: {response.content[:100]}...")
        answers = _parse_answers_payload(response.content, len(questions))
        print(f"[ANSWER_Q] Successfully parsed {len(answers)} answers")
        return answers
    except Exception as e:
        print(f"[ANSWER_Q] Error: {e}")
        if "quota exceeded" in str(e).lower() or "limit reached" in str(e).lower():
            print("[ANSWER_Q] Re-raising quota error")
            raise
        # Return a fallback response
        print("[ANSWER_Q] Generating fallback answers")
        fallback_answers = []
        for i, question in enumerate(questions):
            if any(keyword.lower() in " ".join(observations).lower() for keyword in question.lower().split() if len(keyword) > 3):
                fallback_answers.append("Based on available observations, this aspect requires further inspection.")
            else:
                fallback_answers.append("Not visible in the image")
        
        print(f"[ANSWER_Q] Generated {len(fallback_answers)} fallback answers")
        return fallback_answers


def _parse_answers_payload(raw_text: str, expected_len: int) -> list[str]:
    answers = ["Not visible in the image"] * expected_len
    text = (raw_text or "").strip()
    if not text:
        return answers

    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        text = text[start:end].strip() if end > start else text.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(text)
        payload = data.get("answers", [])
        for i, v in enumerate(payload[:expected_len]):
            answers[i] = (v or "Not visible in the image").strip()
    except Exception:
        pass
    return answers


async def classify_relevant_question_indices(
    image_bytes: bytes, mime_type: str, questions: list[str], observations: list[str]
) -> list[int]:
    print(f"[CLASSIFY] Starting question classification for {len(questions)} questions with {len(observations)} observations")
    question_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    obs_text = "\n".join(observations)
    print(f"[CLASSIFY] Question text length: {len(question_text)}, Observations text length: {len(obs_text)}")
    prompt = f"""
You are helping route inspection questions for a single image.

Task:
- Select only question numbers that are reasonably answerable from this image.
- Prefer precision over recall: include only when visible context is present.
- For unrelated questions (different room/system not shown), exclude them.

Return ONLY valid JSON in this format:
{{
  "relevant_question_numbers": [1, 2, 10]
}}

Observations:
{obs_text}

Questions:
{question_text}
"""
    print("[CLASSIFY] Building image message for classification")
    message = build_image_message(prompt, image_bytes, mime_type)
    try:
        print("[CLASSIFY] Calling LLM for question classification")
        response = await safe_llm_invoke([message])
        text = (response.content or "").strip()
        print(f"[CLASSIFY] Got classification response: {text[:100]}...")
        if "```" in text:
            text = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        nums = data.get("relevant_question_numbers", []) or []
        print(f"[CLASSIFY] Relevant question numbers: {nums}")
        indices = []
        for n in nums:
            try:
                idx = int(n) - 1
                if 0 <= idx < len(questions):
                    indices.append(idx)
            except Exception:
                continue
        # De-duplicate while preserving order
        dedup = []
        seen = set()
        for idx in indices:
            if idx not in seen:
                dedup.append(idx)
                seen.add(idx)
        print(f"[CLASSIFY] Final relevant indices: {dedup}")
        return dedup
    except Exception as e:
        print(f"[CLASSIFY] Error: {e}")
        if "quota exceeded" in str(e).lower() or "limit reached" in str(e).lower():
            print("[CLASSIFY] Re-raising quota error")
            raise
        # Fallback to all questions if classification fails.
        print("[CLASSIFY] Using fallback: all questions")
        return list(range(len(questions)))


async def observe_property_image(image_bytes: bytes, mime_type: str):
    print(f"[OBSERVE] Starting property image observation, size: {len(image_bytes)} bytes, mime: {mime_type}")
    
    observations = await extract_observations(image_bytes, mime_type)
    
    print(f"[OBSERVE] Completed observation extraction, got {len(observations)} observations")
    return observations


async def analyze_property_image(observations: list[str], image_bytes: bytes, mime_type: str, questions):
    print(f"[ANALYZE] Starting property image analysis with {len(observations)} observations and {len(questions)} questions")
    
    relevant_indices = await classify_relevant_question_indices(
        image_bytes, mime_type, questions, observations
    )
    scoped_questions = [questions[i] for i in relevant_indices] if relevant_indices else []
    print(f"[ANALYZE] Found {len(relevant_indices)} relevant questions out of {len(questions)} total")

    if scoped_questions:
        print(f"[ANALYZE] Answering {len(scoped_questions)} scoped questions")
        scoped_answers = await answer_questions(
            image_bytes,
            mime_type,
            scoped_questions,
            observations
        )
    else:
        print("[ANALYZE] No scoped questions to answer")
        scoped_answers = []

    full_answers = ["Not visible in the image"] * len(questions)
    for pos, idx in enumerate(relevant_indices):
        full_answers[idx] = scoped_answers[pos] if pos < len(scoped_answers) else "Not visible in the image"
    
    print(f"[ANALYZE] Mapped answers to {len(full_answers)} total questions")
    answers = json.dumps({"answers": full_answers})

    result = {
        "observations": observations,
        "answers": answers
    }
    print(f"[ANALYZE] Completed property image analysis")
    return result