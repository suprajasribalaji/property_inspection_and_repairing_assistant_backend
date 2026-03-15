from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import os
import base64
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY
)

async def analyze_property_image(image_bytes: bytes, mime_type: str, questions: list[str]):

    question_text = "\n".join(
        [f"{i+1}. {q}" for i, q in enumerate(questions)]
    )

    prompt = f"""
        You are an AI property inspection assistant.

        Your task is to analyze the provided property image and answer the inspection questions listed below.

        IMPORTANT RULES:

        1. Only use information that is directly visible in the image.
        2. Do NOT guess or assume anything that is not visible.
        3. Do NOT use external knowledge or speculation.
        4. If the answer cannot be determined from the image, respond with:
        "Sorry, not visible in the image".
        5. Keep answers short, clear, and written in simple English.
        6. Do not include technical jargon unless it is clearly visible in the image.
        7. Each question must receive exactly one answer.

        QUESTIONS:

        {question_text}

        Return JSON format.
    """

    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    image_url = f"data:{mime_type};base64,{b64_image}"

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
            },
        ]
    )

    response = llm.invoke([message])

    return response.content