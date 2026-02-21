from typing import Dict, List, Union

from pydantic import BaseModel, Field
from llama_cpp import Llama

optimization_log: List[Dict[str, Union[str, float]]] = []

class QuantizationRecommendation(BaseModel):
    """Final recommendation for the best quantization level"""

    best_quantization: str = Field(description="Just the name of the quantization level, no other text, for example: 'q4_0'")

def record_attempt(quantization: str, score: float, observations: str):
    """
    Stores the result of a quantization test in the log.
    """
    entry: Dict[str, Union[str, float]] = {
        "quant": quantization,
        "score": score,
        "notes": observations,
    }
    optimization_log.append(entry)
    return f"Logged attempt {len(optimization_log)}. History now contains {len(optimization_log)} entries."

def get_optimization_history():
    """
    Returns the full list of all previous quantization attempts and their results.
    """
    if not optimization_log:
        return "No attempts have been recorded yet."
    return optimization_log

def run_llama_inference(model_path: str, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
    """Run a single completion against a local llama.cpp model."""
    llm = Llama(model_path=model_path, n_ctx=4096)
    result = llm(prompt, max_tokens=max_tokens, temperature=temperature)
    return result["choices"][0]["text"]
