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
MAX_REQUESTS_PER_DAY = 15  # Stay under the 20 limit
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
    global last_api_call_time, api_calls_today, api_calls_reset_time
    
    now = datetime.now()
    
    # Reset counter if we've passed the reset time
    if now >= api_calls_reset_time:
        api_calls_today = 0
        api_calls_reset_time = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    
    # Check if we've exceeded daily limit
    if api_calls_today >= MAX_REQUESTS_PER_DAY:
        wait_time = (api_calls_reset_time - now).total_seconds()
        raise Exception(f"Daily API quota exceeded. Please wait {wait_time:.0f} seconds or upgrade your plan.")
    
    # Add delay between calls
    time_since_last_call = now.timestamp() - last_api_call_time
    if time_since_last_call < MIN_DELAY_BETWEEN_CALLS:
        await asyncio.sleep(MIN_DELAY_BETWEEN_CALLS - time_since_last_call)
    
    last_api_call_time = datetime.now().timestamp()
    api_calls_today += 1


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(ChatGoogleGenerativeAIError)
)
async def safe_llm_invoke(messages):
    # Check usage tracker first
    can_make, reason = usage_tracker.can_make_request()
    if not can_make:
        raise Exception(f"API usage limit reached: {reason}")
    
    await check_rate_limit()
    
    start_time = time.time()
    try:
        response = await llm.ainvoke(messages)
        response_time = time.time() - start_time
        usage_tracker.log_api_call("gemini_api", True, response_time)
        return response
    except ChatGoogleGenerativeAIError as e:
        response_time = time.time() - start_time
        usage_tracker.log_api_call("gemini_api", False, response_time)
        
        if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
            # Handle quota exceeded specifically
            raise Exception("API quota exceeded. Please try again later or upgrade to a paid plan.")
        else:
            # Re-raise for retry logic
            raise
    except Exception as e:
        response_time = time.time() - start_time
        usage_tracker.log_api_call("gemini_api", False, response_time)
        raise


async def extract_observations(image_bytes: bytes, mime_type: str):

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

    message = build_image_message(prompt, image_bytes, mime_type)

    try:
        response = await safe_llm_invoke([message])
        data = json.loads(response.content)
        return data["observations"]
    except Exception as e:
        print(f"Error in extract_observations: {e}")
        if "quota exceeded" in str(e).lower() or "limit reached" in str(e).lower():
            raise
        return []


async def answer_questions(image_bytes: bytes, mime_type: str, questions, observations):

    question_text = "\n".join(
        [f"{i+1}. {q}" for i, q in enumerate(questions)]
    )

    obs_text = "\n".join(observations)

    prompt = f"""

        You are an AI property inspection assistant.

        Your task is to analyze a property image and answer inspection questions using the visible information in the image and provided observations.
        
        ANSWERING GUIDELINES:

            1. Use information that is explicitly visible in the image or mentioned in observations

            2. For relevant questions (about what's visible in the image):
                - Provide specific, detailed answers based on what you can see
                - Describe the condition, materials, any issues visible
                - Be helpful and informative

            3. For partially relevant questions:
                - Answer what you can observe
                - Acknowledge limitations if you can't see the full picture

            4. For completely unrelated questions (different room/system not shown):
                - Respond with: "Not visible in the image"

            5. Answer style:
            - Keep answers concise but detailed
            - Use simple, clear English
            - Be specific about what you observe
            - Mention materials, conditions, visible issues

            6. Return ONLY valid JSON in this format:

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

            IMPORTANT: Look carefully at the image and provide helpful answers for anything that's visible. Don't default to "Not visible in the image" unless the question is truly unrelated to what's shown.
    """

    message = build_image_message(prompt, image_bytes, mime_type)

    try:
        response = await safe_llm_invoke([message])
        return _parse_answers_payload(response.content, len(questions))
    except Exception as e:
        print(f"Error in answer_questions: {e}")
        if "quota exceeded" in str(e).lower() or "limit reached" in str(e).lower():
            raise
        # Return a fallback response
        fallback_answers = []
        for i, question in enumerate(questions):
            if any(keyword.lower() in " ".join(observations).lower() for keyword in question.lower().split() if len(keyword) > 3):
                fallback_answers.append("Based on available observations, this aspect requires further inspection.")
            else:
                fallback_answers.append("Not visible in the image")
        
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
    question_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    obs_text = "\n".join(observations)
    prompt = f"""
You are helping route inspection questions for a single image.

Task:
- Select question numbers that could be answered based on what's visible in this image
- Include questions about visible materials, conditions, fixtures, or any observable features
- Be inclusive: if you can see part of a system or related components, include those questions
- Only exclude questions that are completely unrelated (e.g., asking about roof when showing plumbing)

Return ONLY valid JSON in this format:
{{
  "relevant_question_numbers": [1, 2, 10]
}}

Observations:
{obs_text}

Questions:
{question_text}

Guideline: Include more questions rather than fewer - let the answering stage determine what can actually be answered.
"""
    message = build_image_message(prompt, image_bytes, mime_type)
    try:
        response = await safe_llm_invoke([message])
        text = (response.content or "").strip()
        if "```" in text:
            text = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        nums = data.get("relevant_question_numbers", []) or []
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
        return dedup
    except Exception as e:
        print(f"Error in classify_relevant_question_indices: {e}")
        if "quota exceeded" in str(e).lower() or "limit reached" in str(e).lower():
            raise
        # Fallback to all questions if classification fails.
        return list(range(len(questions)))


async def observe_property_image(image_bytes: bytes, mime_type: str):

    observations = await extract_observations(image_bytes, mime_type)

    return observations


async def analyze_property_image(observations: list[str], image_bytes: bytes, mime_type: str, questions):
    relevant_indices = await classify_relevant_question_indices(
        image_bytes, mime_type, questions, observations
    )
    scoped_questions = [questions[i] for i in relevant_indices] if relevant_indices else []

    if scoped_questions:
        scoped_answers = await answer_questions(
            image_bytes,
            mime_type,
            scoped_questions,
            observations
        )
    else:
        scoped_answers = []

    full_answers = ["Not visible in the image"] * len(questions)
    for pos, idx in enumerate(relevant_indices):
        full_answers[idx] = scoped_answers[pos] if pos < len(scoped_answers) else "Not visible in the image"

    answers = json.dumps({"answers": full_answers})

    return {
        "observations": observations,
        "answers": answers
    }