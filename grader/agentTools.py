from pydantic import BaseModel, Field, ConfigDict
import logging

class QuantizationRecommendation(BaseModel):
    """Final recommendation for the next best-fit model name."""
    """The alias feild should always remain the litteral string "model name" its functionality is strictly for readability"""

    model_config = ConfigDict(populate_by_name=True)

    model_name: str = Field(
        alias="model name",
        description="Just the model name string, no other text.",
    )   

    def outputThinking(reasoning : str){
        """Takes in reasoning for choosing model as a string, and console logs "reasoning" if DEBUG env variable is set to DEBUG"""
        """Should always call this befor running to completion."""

        logging.debug(f"OpenAI LLM reasoning: {reasoning}")
    }


    
