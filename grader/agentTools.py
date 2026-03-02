from pydantic import BaseModel, Field, ConfigDict
from agents import function_tool
import logging

logger = logging.getLogger(__name__)


class QuantizationRecommendation(BaseModel):
    """Recommended model configuration plus a quality score for the latest inference."""

    model_config = ConfigDict(populate_by_name=True)

    model_name: str = Field(
        alias="model name",
        description="Recommended model filename.",
    )
    max_tokens: int = Field(
        alias="max tokens",
        ge=1,
        description="Recommended max output tokens.",
    )
    temperature: float = Field(
        ge=0.0,
        description="Recommended temperature.",
    )
    top_p: float = Field(
        alias="top p",
        ge=0.0,
        le=1.0,
        description="Recommended nucleus sampling top_p.",
    )
    top_k: int = Field(
        alias="top k",
        ge=0,
        description="Recommended top_k value.",
    )
    repeat_penalty: float = Field(
        alias="repeat penalty",
        ge=0.0,
        description="Recommended repetition penalty.",
    )
    quality_score: int = Field(
        alias="quality score",
        ge=1,
        le=100,
        description="Quality score from 1 to 100 for the latest inference.",
    )


@function_tool
def outputThinking(reasoning: str) -> None:
    """Log OpenAI LLM reasoning when debug logging is enabled."""
    logger.debug("OpenAI LLM reasoning: %s", reasoning)
