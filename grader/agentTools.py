from pydantic import BaseModel, Field
from agents import function_tool, RunContextWrapper



class QuantizationRecommendation(BaseModel):
    """Final recommendation for the best quantization level"""

    best_quantization: str = Field(description="Just the name of the quantization level, no other text, for example: 'q4_0'")


    