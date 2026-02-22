import os
from openai import OpenAI
from dotenv import load_dotenv
from agents import Agent, Runner, Tool
import prompts

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

model = "gpt-4o"
temperature = 0.0
max_tokens = 1000

system_prompt = prompts.system_prompt
prompt = prompts.generate_prompt()

def grader_results():
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content

agent = Agent(
    name="Grader",
    system_prompt=system_prompt,
    model=model,
    tools=[],
)
