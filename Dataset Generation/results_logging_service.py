import json
import logging

from shared_types import RecommendationRecord, RowResult

logger = logging.getLogger(__name__)


class ResultsLoggingService:
    """Stores the final conversation results for every dataset row processed."""

    result_store: list[dict]
    row_number:   int
    log_file:     str

    def __init__(self, log_file: str = "results.log") -> None:
        self.result_store = []
        self.row_number   = 0
        self.log_file     = log_file

    def record_result(self, result: RowResult, recommendation: RecommendationRecord | None = None) -> None:
        entry = {
            "result": result,
            "recommendation": recommendation,
        }
        self.result_store.append(entry)
        self.row_number += 1
        # Print the model that produced this result for real-time visibility
        model_entry = next((e.get("model") for e in result if isinstance(e, dict) and "model" in e), "unknown")
        logger.debug("[Row %s] Saved result. True model: %s", self.row_number, model_entry)
        self.flush_latest()

    def flush_latest(self) -> None:
        """Append the most recently recorded result to the log file."""
        if not self.result_store:
            return
        latest = self.result_store[-1]
        with open(self.log_file, "a") as f:
            f.write(f"--- Row {self.row_number} ---\n")
            f.write(json.dumps(latest, indent=2))
            f.write("\n\n")

    def print_results(self) -> None:
        for result in self.result_store:
            print(result)
