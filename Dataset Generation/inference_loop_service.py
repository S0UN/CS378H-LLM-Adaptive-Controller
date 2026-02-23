import subprocess
import os
import requests
from datasets import DatasetDict
from dataset import DatasetService
from model_manager import ModelDownloader
from logging_service import LoggingService
# Type aliases for clarity
Turn      = dict[str, str]           # {"input": ..., "output": ...} or {"from": ..., "value": ...}
TurnResult = dict[str, str | None]   # {"input": ..., "expected_output": ..., "inference_output": ...}
Message   = dict[str, str]           # {"role": ..., "content": ...}

class InferenceLoopService:

    dataset:          DatasetDict
    model_name:       str
    model_downloader: ModelDownloader
    compose_dir:      str
    logging_service: LoggingService
    result_store : ResultsLoggingService

    def __init__(
        self,
        dataset_name:     str,
        model_name:       str,
        model_downloader: ModelDownloader,
        compose_dir:      str = "..",
    ) -> None:
        """
        Args:
            dataset_name:     Key from DatasetService.url_map (e.g. "Capybara")
            model_name:       GGUF filename to load (e.g. "llama-2-7b.Q4_K_M.gguf")
            model_downloader: Configured ModelDownloader instance
            compose_dir:      Path to the directory containing docker-compose.yml
        """
        self.dataset          = DatasetService().get_database(dataset_name)
        self.model_name       = model_name
        self.model_downloader = model_downloader
        self.compose_dir      = compose_dir
        self.logging_service = LoggingService()
        self.result_store = ResultsLoggingService()
        
        # Download the model if not already cached
        self.model_downloader.download(model_name)

    def run(self) -> ResultsLoggingService:
        """Load the model into Docker then run the full inference loop over the dataset."""
        self.load_model()

        server_url  = os.environ.get("MODEL_BASE_URL", "http://localhost:8080")

        split = "train" if "train" in self.dataset else list(self.dataset.keys())[0]

        for row in self.dataset[split]:
            conversation: list[Turn] = row.get("conversation", row.get("conversations", []))
            results = self.run_conversation(conversation, server_url)
            record: Dict[str, Union[str, float]] = #openAI call
            self.logging_service.record_attempt(suggested_model)

            while not found_optimal_model():
                self.load_model(suggested_model)
                results = self.run_conversation(conversation,server_url)
                record: Dict[str, Union[str, float]] = #openAI call
                self.logging_service.record_attempt(record)
            
            self.logging_service.clear()

            self.result_store.record_result(results)
            

        return self.result_store

    def found_optimal_model(self) -> bool:
      log = self.logging_service.get_optimization_history()

      if not isinstance(log, list) or len(log) < 2:
          return False

      quant_n = log[-1].get("quant")
      quant_n_minus_1 = log[-2].get("quant")

      # Back-to-back repeat
      if quant_n == quant_n_minus_1:
          return True

      # Repeat at n and n-2
      if len(log) >= 3:
          quant_n_minus_2 = log[-3].get("quant")
          if quant_n == quant_n_minus_2:
              return True

      return False



    def run_conversation(self, conversation: list[Turn], server_url: str) -> list[TurnResult]:
        """Run a multi-turn conversation, feeding the model's own output back as context each turn."""
        results: list[TurnResult] = []
        history: list[Message]    = []
        system_prompt: str        = ""

        for turn in conversation:
            # Normalize turn format
            if "input" in turn:
                # Capybara format: every turn is a user/assistant pair
                user_msg = turn["input"]
                expected = turn["output"]
            else:
                # OpenHermes format: turns have explicit roles (system/human/gpt)
                role = turn.get("from")

                if role == "system":
                    # Capture system prompt from dataset and inject into history
                    system_prompt = turn["value"]
                    history.append({"role": "system", "content": system_prompt})
                    continue

                if role == "gpt":
                    # Skip gold assistant turns - we generate these ourselves
                    continue

                # role == "human"
                user_msg = turn["value"]
                expected = ""   # gpt reply is a separate sibling turn in the list

            # Append the user message to the running history
            history.append({"role": "user", "content": user_msg})

            # Send the full history to the model
            inference_output: str | None = None
            try:
                response = requests.post(
                    f"{server_url}/v1/chat/completions",
                    json={"messages": history},
                    timeout=10000,
                )
                inference_output = response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"Inference error on turn '{user_msg[:60]}...': {e}")

            results.append({
                "input":            user_msg,
                "expected_output":  expected,
                "inference_output": inference_output,
            })

            # Option A: feed the model's own reply back as context for the next turn
            history.append({"role": "assistant", "content": inference_output or ""})

        # Prepend system prompt (from dataset) and append model name as metadata entries
        return (
            [{"system_prompt": system_prompt}] +
            results +
            [{"model": self.model_name}]
        )

    def load_model(self) -> None:
        """Start the model container, restarting it only if a different model is running."""
        running = self.get_running_model()

        if running is not None:
            running_filename = os.path.basename(running)

            if running_filename == self.model_name:
                print(f"Model already running: {self.model_name}")
                return

            # A different model is loaded — stop it first
            print(f"Stopping current model: {running_filename}")
            subprocess.run(
                ["docker-compose", "stop", "model"],
                cwd=self.compose_dir,
                check=True,
            )

        # Start the container with the target model via MODEL_FILE env var
        print(f"Starting model: {self.model_name}")
        env = {**os.environ, "MODEL_FILE": self.model_name}
        subprocess.run(
            ["docker-compose", "up", "-d", "--force-recreate", "model"],
            cwd=self.compose_dir,
            env=env,
            check=True,
        )

    def get_running_model(self) -> str | None:
        """
        Query the llama.cpp server for the currently loaded model.

        Returns:
            The model path string (e.g. "/models/foo.Q4_K_M.gguf"),
            or None if the container is not running.
        """
        try:
            response = requests.get("http://localhost:8080/v1/models", timeout=2)
            models: list[dict] = response.json()["data"]
            return models[0]["id"] if models else None
        except Exception:
            return None  # container not running
