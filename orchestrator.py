"""
Orchestrator — runs the full inference + optimization loop end-to-end.

Usage:
    python orchestrator.py

Environment variables (can also be set in a .env file):
    DATASET_NAME      Dataset shortcut (default: "Capybara")
    MODEL_REPO        HuggingFace repo ID or MODEL_REPOS shortcut (default: LLAMA2_7B)
    MODEL_QUANT       Quantization shortcut or level string (default: Q4_K_M)
    MODEL_CACHE_DIR   Where to store downloaded GGUF files (default: ./models)
    OPENAI_API_KEY    Required for the GPT-4o grader agent
"""
import os
import sys

from dotenv import load_dotenv

# ── sys.path setup ────────────────────────────────────────────────────────────
# Project root must be first so `grader/` is importable as a package.
# `Dataset Generation/` is added so the flat modules inside it are importable.
_project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _project_root)                                          # for grader.*
sys.path.insert(0, os.path.join(_project_root, "Dataset Generation"))     # for inference_loop_service, etc.

from model_config import DEFAULT_MODEL, DEFAULT_QUANT, DEFAULT_CACHE_DIR  # noqa: E402
from model_manager import ModelDownloader                                  # noqa: E402
from inference_loop_service import InferenceLoopService                    # noqa: E402


def main() -> None:
    load_dotenv()

    # Support both OPENAI_API_KEY (standard) and OPENAI_KEY (legacy alias)
    if not os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_KEY"):
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")  # type: ignore[arg-type]

    dataset_name = os.getenv("DATASET_NAME", "Capybara")
    model_repo   = os.getenv("MODEL_REPO",   DEFAULT_MODEL)
    quant        = os.getenv("MODEL_QUANT",  DEFAULT_QUANT)
    cache_dir    = os.getenv("MODEL_CACHE_DIR", DEFAULT_CACHE_DIR)
    compose_dir  = _project_root

    # Download the starting model (LRU cache handles eviction if needed)
    downloader  = ModelDownloader(repo_id=model_repo, cache_dir=cache_dir)
    model_path  = downloader.download(quant)

    if model_path is None:
        raise RuntimeError(
            f"Failed to download model for repo='{model_repo}', quant='{quant}'."
        )

    model_name = os.path.basename(model_path)

    service = InferenceLoopService(
        dataset_name=dataset_name,
        model_name=model_name,
        model_downloader=downloader,
        compose_dir=compose_dir,
    )

    results = service.run()
    results.print_results()


if __name__ == "__main__":
    main()
