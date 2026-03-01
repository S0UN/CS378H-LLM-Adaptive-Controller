from pydantic import BaseModel, Field, ConfigDict
import logging

logger = logging.getLogger(__name__)

class QuantizationRecommendation(BaseModel):
    """Final recommendation for the next best-fit model name."""
    """The alias feild should always remain the litteral string "model name" its functionality is strictly for readability"""

    model_config = ConfigDict(populate_by_name=True)

    model_name: str = Field(
        alias="model name",
        description="Just the model name string, no other text.",
    )


def outputThinking(reasoning: str) -> None:
    """Log OpenAI LLM reasoning when debug logging is enabled."""
    logger.debug("OpenAI LLM reasoning: %s", reasoning)
