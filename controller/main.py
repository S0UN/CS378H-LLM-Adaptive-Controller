import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 200
    temperature: float = 0.0
    model: str = "gpt-4o-mini"  # Default model, can be changed to gpt-4, gpt-3.5-turbo, etc.

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat")
async def chat(req: ChatRequest):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
    
    max_tokens = min(req.max_tokens, 4096)  # OpenAI supports more tokens
    
    try:
        response = await client.chat.completions.create(
            model=req.model,
            messages=[
                {"role": "user", "content": req.prompt}
            ],
            max_tokens=max_tokens,
            temperature=req.temperature,
        )
        
        text = response.choices[0].message.content
        
        return {
            "text": text,
            "raw": {
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

