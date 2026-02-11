import os
import httpx
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_BASE_URL = os.getenv("MODEL_BASE_URL", "http://localhost:8080")

app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 200
    temperature: float = 0.0

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat")
async def chat(req: ChatRequest):
    max_tokens = min(req.max_tokens, 200)

    payload = {
        "prompt": req.prompt,
        "n_predict": max_tokens,
        "temperature": req.temperature,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(f"{MODEL_BASE_URL}/completion", json=payload)
        r.raise_for_status()
        data = r.json()

    text = data.get("content") if data.get("content") is not None else data.get("completion", "")

    return {"text": text, "raw": data}

