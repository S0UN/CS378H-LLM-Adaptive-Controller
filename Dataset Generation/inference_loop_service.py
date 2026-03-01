import subprocess
import os
import time
import requests
import logging
from datasets import DatasetDict          # HuggingFace datasets library (was wrongly "from dataset import DatasetDict")
from dataset import DatasetService
from model_manager import ModelDownloader
from logging_service import LoggingService
from results_logging_service import ResultsLoggingService
from grader.agent import GraderService

logger = logging.getLogger(__name__)

# Type aliases for clarity
Turn       = dict[str, str]           # {"input": ..., "output": ...} or {"from": ..., "value": ...}
TurnResult = dict[str, str | None]    # {"input": ..., "expected_output": ..., "inference_output": ...}
Message    = dict[str, str]           # {"role": ..., "content": ...}


class InferenceLoopService:

    dataset:          DatasetDict
    model_name:       str
    model_downloader: ModelDownloader
    compose_dir:      str
    logging_service:  LoggingService
    result_store:     ResultsLoggingService
    grader:           GraderService
    model_list:       list[str]

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
        self.logging_service  = LoggingService()
        self.result_store     = ResultsLoggingService()
        self.grader           = GraderService()

        # Populate the list of available model filenames from the HuggingFace repo
        # so the grader can reference them when recommending the next quantization.
        self.model_list = set(self.model_downloader.list_available())


        # Download the starting model if not already cached
        self.model_downloader.download(model_name)

    def _wait_for_model_ready(self, timeout: int = 3600, poll_interval: int = 3) -> None:
        """Block until the llama.cpp server responds, or raise after timeout seconds."""
        print(f"Waiting for model server to be ready (up to {timeout}s)...")
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.get_running_model() is not None:
                print("Model server is ready.")
                return
            time.sleep(poll_interval)
        raise TimeoutError(
            f"Model server did not become ready within {timeout} seconds."
        )

    def run(self) -> ResultsLoggingService:
        """Load the model into Docker then run the full inference loop over the dataset."""
        self.load_model()
        self._wait_for_model_ready()

        server_url = os.environ.get("MODEL_BASE_URL", "http://localhost:8080")

        split = "train" if "train" in self.dataset else list(self.dataset.keys())[0]
        total_rows = len(self.dataset[split])
        logger.debug("%s", "=" * 60)
        logger.debug("Starting inference loop: %s rows, split='%s'", total_rows, split)
        logger.debug("%s", "=" * 60)

        for row_idx, row in enumerate(self.dataset[split], start=1):
            conversation: list[Turn] = row.get("conversation", row.get("conversations", []))
            num_turns = len(conversation)
            logger.debug("[Row %s/%s] Starting - %s turns in conversation", row_idx, total_rows, num_turns)

            # ── First inference pass with the starting model ──────────────────
            logger.debug("[Row %s] Running first inference pass with model: %s", row_idx, self.model_name)
            results = self.run_conversation(conversation, server_url)
            logger.debug("[Row %s] Inference done. Calling grader...", row_idx)
            record  = self.grader.run(
                optimization_log=self.logging_service.get_optimization_history(),
                last_inference=results,
                model_names=self.model_list
            )
            suggested = record.get("model name", "unknown")
            logger.debug("[Row %s] Grader suggested: %s", row_idx, suggested)
            self.logging_service.record_attempt(record)

            # ── Optimization loop ─────────────────────────────────────────────
            opt_iter = 0
            retry_count : int = 0
            while not self.found_optimal_model():
                if (retry_count > 3):
                    raise Exception("Failed to find optimal model after 3 retries")

                opt_iter += 1
                recent: str = self.logging_service.get_most_recent_suggestion()
                logger.debug("[Row %s] Opt iter %s: switching to model '%s'", row_idx, opt_iter, recent)
                self.load_model(recent)
                results = self.run_conversation(conversation, server_url)
                logger.debug("[Row %s] Opt iter %s: calling grader...", row_idx, opt_iter)
                record  = self.grader.run(
                    optimization_log=self.logging_service.get_optimization_history(),
                    last_inference=results,
                    model_names=self.model_list,
                )

                # Check if the suggested model is valid
                suggested = record.get("model name", "")
                is_valid =  suggested in self.model_list
                if (not is_valid):
                    retry_count += 1
                    continue # skip to next iteration

                suggested = record.get("model name", "unknown")
                logger.debug("[Row %s] Opt iter %s: grader suggested: %s", row_idx, opt_iter, suggested)
                self.logging_service.record_attempt(record)

            logger.debug(
                "[Row %s] Converged after %s optimization iterations. Final model: %s",
                row_idx,
                opt_iter,
                self.model_name,
            )
            self.logging_service.clear()
            self.result_store.record_result(results)

        logger.debug("%s", "=" * 60)
        logger.debug("Inference loop complete. %s rows processed.", total_rows)
        logger.debug("%s", "=" * 60)
        return self.result_store

    def found_optimal_model(self) -> bool:
        log = self.logging_service.get_optimization_history()

        if not isinstance(log, list) or len(log) < 2:
            return False

        # Records are dicts like {"model name": "llama-2-7b-chat.Q4_K_M.gguf"}
        # (was incorrectly looking up "quant" key)
        key = "model name"
        model_n       = log[-1].get(key)
        model_n_minus1 = log[-2].get(key)

        # Back-to-back repeat → converged
        if model_n == model_n_minus1:
            return True

        # Oscillating between two levels (n == n-2) → converged
        if len(log) >= 3:
            model_n_minus2 = log[-3].get(key)
            if model_n == model_n_minus2:
                return True

        return False

    def run_conversation(self, conversation: list[Turn], server_url: str) -> list[TurnResult]:
        """Run a multi-turn conversation, feeding the model's own output back as context each turn."""
        results: list[TurnResult] = []
        history: list[Message]    = []
        system_prompt: str        = ""
        turn_num = 0

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
                    # Skip gold assistant turns — we generate these ourselves
                    continue

                # role == "human"
                user_msg = turn["value"]
                expected = ""   # gpt reply is a separate sibling turn in the list

            # Append the user message to the running history
            history.append({"role": "user", "content": user_msg})
            turn_num += 1
            print(f"  Turn {turn_num}: sending '{user_msg[:60]}{'...' if len(user_msg) > 60 else ''}'")

            # Send the full history to the model
            inference_output: str | None = None
            try:
                response = requests.post(
                    f"{server_url}/v1/chat/completions",
                    json={
                        "messages":   history,
                        "max_tokens": 512,
                        "stop":       ["<|im_end|>", "<|eot_id|>", "</s>"],
                    },
                    timeout=10000,
                )
                inference_output = response.json()["choices"][0]["message"]["content"]
                print(f"  Turn {turn_num}: got response ({len(inference_output)} chars)")
            except Exception as e:
                print(f"  Turn {turn_num}: inference error — {e}")

            results.append({
                "input":            user_msg,
                "expected_output":  expected,
                "inference_output": inference_output,
            })

            # Option A: feed the model's own reply back as context for the next turn
            history.append({"role": "assistant", "content": inference_output or ""})

        # Prepend system prompt (from dataset) and append model name as metadata entries
        return (
            [{"system_prompt": system_prompt}]
            + results
            + [{"model": self.model_name}]
        )

    def load_model(self, model_name: str | None = None) -> None:
        """Start the model container, restarting it only if a different model is running.

        Args:
            model_name: GGUF filename to load.  Defaults to self.model_name when None.
        """
        target  = model_name if model_name is not None else self.model_name
        self.model_name = target
        running = self.get_running_model()

        if running is not None:
            running_filename = os.path.basename(running)

            if running_filename == target:
                print(f"Model already running: {target}")
                return

            # A different model is loaded — stop it first
            print(f"Stopping current model: {running_filename}")
            subprocess.run(
                ["docker-compose", "stop", "model"],
                cwd=self.compose_dir,
                check=True,
            )

        # Ensure the target model is downloaded before starting the container
        if target != self.model_name:
            self.model_downloader.download(target)

        # Start the container with the target model via MODEL_FILE env var
        print(f"Starting model: {target}")
        env = {**os.environ, "MODEL_FILE": target}
        subprocess.run(
            ["docker-compose", "up", "-d", "--force-recreate", "model"],
            cwd=self.compose_dir,
            env=env,
            check=True,
        )
        self._wait_for_model_ready()

    def get_running_model(self) -> str | None:
        """Query the llama.cpp server for the currently loaded model.

        Returns:
            The model path string (e.g. "/models/foo.Q4_K_M.gguf"),
            or None if the container is not running.
        """
        try:
            response = requests.get("http://localhost:8080/v1/models", timeout=2)
            models: list[dict] = response.json()["data"]
            return models[0]["id"] if models else None
        except Exception:
            return None   # container not running
