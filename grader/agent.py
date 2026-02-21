from agents import Agent, Runner
from dotenv import load_dotenv

import agentTools
import prompts

load_dotenv()

agent = Agent(
    name="Grader",
    instructions=prompts.build_quantization_instructions,
    model="gpt-4o",
    tools=[agentTools.record_attempt, agentTools.get_optimization_history],
    output_type=agentTools.QuantizationRecommendation,
)

async def agent_runner():
    result = await Runner.run(
        agent,
        "Start the optimization loop",
        context={"gold" : "populate after chotes is done", "first_iteration" : "populate after chotes is done"}
    )
    print(result.final_output)
