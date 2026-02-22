import importlib.util
import pathlib
import sys
import types
import unittest


class _FakeAgentClass:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeRunnerClass:
    @staticmethod
    async def run(*args, **kwargs):
        return None


class _FakeRecommendation:
    pass


def _fake_build_quantization_instructions(context, agent):
    return "instructions"


def _fake_record_attempt(*args, **kwargs):
    return "ok"


def _fake_get_optimization_history():
    return []


class AgentModuleTests(unittest.TestCase):
    def setUp(self):
        self._saved_modules = {
            name: sys.modules.get(name)
            for name in ["agents", "dotenv", "prompts", "agentTools", "grader_agent_under_test"]
        }

        fake_agents = types.ModuleType("agents")
        fake_agents.Agent = _FakeAgentClass
        fake_agents.Runner = _FakeRunnerClass
        sys.modules["agents"] = fake_agents

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        sys.modules["dotenv"] = fake_dotenv

        fake_prompts = types.ModuleType("prompts")
        fake_prompts.build_quantization_instructions = _fake_build_quantization_instructions
        sys.modules["prompts"] = fake_prompts

        fake_agent_tools = types.ModuleType("agentTools")
        fake_agent_tools.record_attempt = _fake_record_attempt
        fake_agent_tools.get_optimization_history = _fake_get_optimization_history
        fake_agent_tools.QuantizationRecommendation = _FakeRecommendation
        sys.modules["agentTools"] = fake_agent_tools

    def tearDown(self):
        for name, module in self._saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_agent_is_configured_with_expected_fields(self):
        module_path = pathlib.Path("grader/agent.py")
        spec = importlib.util.spec_from_file_location("grader_agent_under_test", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)

        configured_agent = module.agent
        self.assertEqual(configured_agent.kwargs["name"], "Grader")
        self.assertEqual(configured_agent.kwargs["model"], "gpt-4o")
        self.assertIs(configured_agent.kwargs["instructions"], _fake_build_quantization_instructions)
        self.assertEqual(
            configured_agent.kwargs["tools"],
            [_fake_record_attempt, _fake_get_optimization_history],
        )
        self.assertIs(configured_agent.kwargs["output_type"], _FakeRecommendation)


if __name__ == "__main__":
    unittest.main()
