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
    # v0 “Budgeted Router”: trivial policy (you’ll replace this later)
    # Example: cap max_tokens for cheap attempt.
    max_tokens = min(req.max_tokens, 200)

    # llama.cpp server supports OpenAI-compatible routes in many builds;
    # but it also supports a native completion route. To keep this robust,
    # we’ll call the server's completion endpoint commonly exposed by llama.cpp server.
    # If your image exposes only OpenAI routes, swap to /v1/chat/completions.
    payload = {
        "prompt": req.prompt,
        "n_predict": max_tokens,
        "temperature": req.temperature,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(f"{MODEL_BASE_URL}/completion", json=payload)
        r.raise_for_status()
        data = r.json()

    # llama.cpp returns different shapes depending on endpoint/version; this is the common one:
    text = data.get("content") or data.get("completion") or str(data)

    return {"text": text, "raw": data}

