from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.inspect import router as inspect_router
from app.api.chat import router as chat_router
from app.api.usage import router as usage_router
from app.services.database_service import init_db

app = FastAPI()

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    await init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8080", "http://127.0.0.0:8080"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
app.include_router(inspect_router)
app.include_router(chat_router)
app.include_router(usage_router)

@app.get("/")
def main():
    return { "message": "AI inspection API running" }