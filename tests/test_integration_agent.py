import asyncio
import os
import unittest


RUN_INTEGRATION = os.getenv("RUN_INTEGRATION") == "1"


@unittest.skipUnless(RUN_INTEGRATION, "Set RUN_INTEGRATION=1 to run integration tests.")
class AgentIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
        if not api_key:
            raise unittest.SkipTest("OPENAI_API_KEY (or OPENAI_KEY) is required for integration tests.")

        os.environ["OPENAI_API_KEY"] = api_key

        try:
            from agents import Agent, Runner  # pylint: disable=import-outside-toplevel
        except Exception as exc:  # pragma: no cover - environment dependent
            raise unittest.SkipTest(f"agents SDK unavailable: {exc}")

        cls.Agent = Agent
        cls.Runner = Runner
        cls.model = os.getenv("OPENAI_TEST_MODEL", "gpt-4o-mini")

    def test_basic_agent_round_trip(self):
        async def _run():
            agent = self.Agent(
                name="PingAgent",
                instructions="Reply with exactly: PONG",
                model=self.model,
            )
            result = await self.Runner.run(agent, "ping")
            self.assertIn("pong", str(result.final_output).lower())

        asyncio.run(_run())

    def test_context_can_populate_dynamic_instructions(self):
        async def _run():
            token = "CTX-42"

            def dynamic_instructions(run_context, _agent):
                return f"Return exactly this token: {run_context.context['token']}"

            agent = self.Agent(
                name="ContextAgent",
                instructions=dynamic_instructions,
                model=self.model,
            )

            result = await self.Runner.run(
                agent,
                "Please respond now.",
                context={"token": token},
            )
            self.assertIn(token.lower(), str(result.final_output).lower())

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
