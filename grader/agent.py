import asyncio
from agents import Agent, Runner

import agentTools
import prompts


class GraderService:

    def __init__(
        self,
        model: str = "gpt-4o",
        tools: list | None = None,
        output_type=agentTools.QuantizationRecommendation,
    ) -> None:
        if tools is None:
            tools = [agentTools.record_attempt, agentTools.get_optimization_history]

        self.agent = Agent(
            name="Grader",
            instructions=prompts.build_quantization_instructions,
            model=model,
            tools=tools,
            output_type=output_type,
        )

    def run_agent(self, optimization_log=None, last_inference=None, model_names=None) -> dict[str, str] | dict:
        if optimization_log is None:
            optimization_log = []
        if last_inference is None:
            last_inference = []
        if model_names is None:
            model_names = []

        async def _run():
            result = await Runner.run(
                self.agent,
                "Analyze the information given and make your educated guess",
                context={
                    "optimization_log": optimization_log,
                    "last_inference": last_inference,
                    "model_names": model_names
                },
            )
            output = result.final_output
            if hasattr(output, "model_dump"):
                return output.model_dump(by_alias=True)
            return output

        return asyncio.run(_run())
