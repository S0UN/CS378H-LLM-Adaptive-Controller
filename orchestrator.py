"""
Orchestrator to run the full inference loop end-to-end.
"""
import os
import sys

from dotenv import load_dotenv

# Add Dataset Generation to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Dataset Generation"))

from model_config import DEFAULT_MODEL, DEFAULT_QUANT, DEFAULT_CACHE_DIR
from model_manager import ModelDownloader
from inference_loop_service import InferenceLoopService


def main() -> None:
    load_dotenv()

    dataset_name = os.getenv("DATASET_NAME", "Capybara")
    model_repo = os.getenv("MODEL_REPO", DEFAULT_MODEL)
    quant = os.getenv("MODEL_QUANT", DEFAULT_QUANT)
    cache_dir = os.getenv("MODEL_CACHE_DIR", DEFAULT_CACHE_DIR)
    compose_dir = os.path.dirname(__file__)

    downloader = ModelDownloader(repo_id=model_repo, cache_dir=cache_dir)
    model_path = downloader.download(quant)
    if model_path is None:
        raise RuntimeError("Failed to download model for the selected quantization.")

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
