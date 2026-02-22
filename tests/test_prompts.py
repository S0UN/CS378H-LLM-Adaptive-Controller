import importlib
import sys
import types
import unittest


class _FakeRunContextWrapper:
    def __init__(self, context):
        self.context = context


class _FakeAgent:
    pass


class PromptsTests(unittest.TestCase):
    def setUp(self):
        self._old_agents = sys.modules.get("agents")
        fake_agents = types.ModuleType("agents")
        fake_agents.Agent = _FakeAgent
        fake_agents.RunContextWrapper = _FakeRunContextWrapper
        sys.modules["agents"] = fake_agents

        if "grader.prompts" in sys.modules:
            del sys.modules["grader.prompts"]

        self.prompts = importlib.import_module("grader.prompts")

    def tearDown(self):
        if "grader.prompts" in sys.modules:
            del sys.modules["grader.prompts"]

        if self._old_agents is None:
            sys.modules.pop("agents", None)
        else:
            sys.modules["agents"] = self._old_agents

    def test_build_quantization_instructions_includes_context_values(self):
        context = _FakeRunContextWrapper(
            {
                "gold_standard": "ideal output",
                "firstIteration": "first model run",
            }
        )

        instructions = self.prompts.build_quantization_instructions(context, _FakeAgent())

        self.assertIn('GOLD_STANDARD: "ideal output"', instructions)
        self.assertIn('FIRST_ITERATION_SAMPLE: "first model run"', instructions)

    def test_build_quantization_instructions_uses_defaults_when_missing(self):
        context = _FakeRunContextWrapper({})

        instructions = self.prompts.build_quantization_instructions(context, _FakeAgent())

        self.assertIn('GOLD_STANDARD: "No gold standard provided."', instructions)
        self.assertIn('FIRST_ITERATION_SAMPLE: "First inference not provided."', instructions)


if __name__ == "__main__":
    unittest.main()
