"""
Quick smoke test - runs one conversation from Capybara and prints the results.
"""
import sys
import os

# Add Dataset Generation to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Dataset Generation"))

from model_manager import ModelDownloader
from inference_loop_service import InferenceLoopService
from dataset import DatasetService

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME   = "llama-2-7b-chat.Q2_K.gguf"   # change to whatever you downloaded
DATASET_NAME = "Capybara"
COMPOSE_DIR  = os.path.dirname(__file__)       # where docker-compose.yml lives

# ── Run ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # Load one row from the dataset without downloading the whole thing
    print("Loading dataset (first row only)...")
    dataset = DatasetService().get_database(DATASET_NAME)
    split   = "train" if "train" in dataset else list(dataset.keys())[0]
    row     = dataset[split][0]

    conversation = row.get("conversation", row.get("conversations", []))
    print(f"Conversation has {len(conversation)} turns\n")

    # Run the conversation against the already-running llama.cpp server
    # (skips docker-compose management - assumes server is already up)
    from inference_loop_service import InferenceLoopService

    server_url = os.environ.get("MODEL_BASE_URL", "http://localhost:8080")

    downloader = ModelDownloader(cache_dir="./models")
    service    = InferenceLoopService(
        dataset_name     = DATASET_NAME,
        model_name       = MODEL_NAME,
        model_downloader = downloader,
        compose_dir      = COMPOSE_DIR,
    )

    print("Running inference on first conversation...\n")
    results = service.run_conversation(conversation, server_url)

    # ── Print results ─────────────────────────────────────────────────────────
    print("=" * 70)
    for i, turn in enumerate(results, 1):
        print(f"\nTurn {i}")
        print(f"  INPUT:    {turn['input']}")
        print(f"  EXPECTED: {turn['expected_output']}")
        print(f"  MODEL:    {turn['inference_output']}")
        print("-" * 70)

    print(f"\nDone. {len(results)} turns processed.")


if __name__ == "__main__":
    main()
