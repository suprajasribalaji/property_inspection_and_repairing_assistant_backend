from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import os
import base64
import json
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY
)


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


async def extract_observations(image_bytes: bytes, mime_type: str):

    prompt = """
        You are a property observation and extraction AI assistant.

        Look carefully at the property image and list only the visible observations from the image.

        Rules:
        - Only describe what is clearly visible in the image
        - Do not guess
        - Do not assume anything not visible
        - Keep descriptions short

        Return ONLY valid JSON in this format:

        {
            "observations": [
                "observation 1",
                "observation 2"
            ]
        }
    """

    message = build_image_message(prompt, image_bytes, mime_type)

    response = llm.invoke([message])

    try:
        data = json.loads(response.content)
        return data["observations"]
    except:
        return []


async def answer_questions(image_bytes: bytes, mime_type: str, questions, observations):

    question_text = "\n".join(
        [f"{i+1}. {q}" for i, q in enumerate(questions)]
    )

    obs_text = "\n".join(observations)

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
            - short
            - clear
            - in simple English
            - free of technical jargon

            6. Be consistent:
            - Do not change answer style between questions
            - Do not include explanations unless clearly visible

            7. Return ONLY valid JSON in this format:
            - Do not include any additional text or markdown formatting or code blocks.

        Return ONLY valid JSON in this format:
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

    message = build_image_message(prompt, image_bytes, mime_type)

    response = llm.invoke([message])

    return response.content

async def observe_property_image(image_bytes: bytes, mime_type: str):

    observations = await extract_observations(image_bytes, mime_type)

    return observations


async def analyze_property_image(observations: list[str], image_bytes: bytes, mime_type: str, questions):

    answers = await answer_questions(
        image_bytes,
        mime_type,
        questions,
        observations
    )

    return {
        "observations": observations,
        "answers": answers
    }