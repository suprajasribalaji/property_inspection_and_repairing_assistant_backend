from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.inspect import router as inspect_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8080", "http://127.0.0.0:8080"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
app.include_router(inspect_router)

@app.get("/")
def main():
    return { "message": "AI inspection API running" }