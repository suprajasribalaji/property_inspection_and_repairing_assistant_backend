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
        You are a property inspection AI based on the image send by the user.

        Analyze the property image and answer these questions.

        Rules:
        - Only answer based on what is visible in the image
        - If not visible say "Sorry, it's not visible in the image."
        - Answer STRICTLY basend on the image
        - No speculation, hallucination, jargons, or external knowledge
        - Use simple english for the explanation

        Questions:
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