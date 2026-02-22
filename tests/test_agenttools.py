import importlib
import sys
import types
import unittest


class _FakeLlama:
    def __init__(self, model_path, n_ctx):
        self.model_path = model_path
        self.n_ctx = n_ctx

    def __call__(self, prompt, max_tokens, temperature):
        return {
            "choices": [
                {
                    "text": f"model={self.model_path};prompt={prompt};max_tokens={max_tokens};temp={temperature}"
                }
            ]
        }


class AgentToolsTests(unittest.TestCase):
    def setUp(self):
        self._old_llama_cpp = sys.modules.get("llama_cpp")
        fake_llama_cpp = types.ModuleType("llama_cpp")
        fake_llama_cpp.Llama = _FakeLlama
        sys.modules["llama_cpp"] = fake_llama_cpp

        if "grader.agentTools" in sys.modules:
            del sys.modules["grader.agentTools"]

        self.agent_tools = importlib.import_module("grader.agentTools")
        self.agent_tools.optimization_log.clear()

    def tearDown(self):
        if "grader.agentTools" in sys.modules:
            del sys.modules["grader.agentTools"]

        if self._old_llama_cpp is None:
            sys.modules.pop("llama_cpp", None)
        else:
            sys.modules["llama_cpp"] = self._old_llama_cpp

    def test_record_attempt_and_history(self):
        message = self.agent_tools.record_attempt("q4_0", 0.9, "stable and coherent")

        self.assertIn("Logged attempt 1", message)
        history = self.agent_tools.get_optimization_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["quant"], "q4_0")
        self.assertEqual(history[0]["score"], 0.9)

    def test_run_llama_inference_returns_choice_text(self):
        text = self.agent_tools.run_llama_inference(
            model_path="/tmp/mock.gguf",
            prompt="Hello",
            max_tokens=64,
            temperature=0.3,
        )

        self.assertIn("model=/tmp/mock.gguf", text)
        self.assertIn("prompt=Hello", text)
        self.assertIn("max_tokens=64", text)
        self.assertIn("temp=0.3", text)


if __name__ == "__main__":
    unittest.main()
