import asyncio
from agents import Agent, Runner

from grader import agentTools
from grader import prompts


class GraderService:

    def __init__(
        self,
        model: str = "gpt-4o",
        output_type=agentTools.QuantizationRecommendation,
    ) -> None:
        self.agent = Agent(
            name="Grader",
            instructions=prompts.build_quantization_instructions,
            model=model,
            output_type=output_type,
        )

    def run(
        self,
        optimization_log: list | None = None,
        last_inference: list | None = None,
        model_names: list | None = None,
    ) -> dict[str, str]:
        if optimization_log is None:
            optimization_log = []
        if last_inference is None:
            last_inference = []
        if model_names is None:
            model_names = []

        async def _run() -> dict[str, str]:
            result = await Runner.run(
                self.agent,
                "Analyze the information given and make your educated guess",
                context={
                    "optimization_log": optimization_log,
                    "last_inference": last_inference,
                    "model_names": model_names,
                },
            )
            output = result.final_output
            if hasattr(output, "model_dump"):
                return output.model_dump(by_alias=True)
            return output

        return asyncio.run(_run())
