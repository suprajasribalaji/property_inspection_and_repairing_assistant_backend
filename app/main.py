from fastapi import FastAPI
from app.api.inspect import router as inspect_router

app = FastAPI()

app.include_router(inspect_router)

@app.get("/")
def main():
    return { "message": "AI inspection API running" }