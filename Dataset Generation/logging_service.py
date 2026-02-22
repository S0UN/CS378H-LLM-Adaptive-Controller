
from typing import Dict, List, Union

class LoggingService:
    optimization_log: List[Dict[str, Union[str, float]]]

    def __init__(self):
        self.optimization_log = []

    def record_attempt(self, record: Dict[str, Union[str, float]]):
        """
        Stores the result of a quantization test in the log.
        """

        optimization_log.append(record)
        return f"Logged attempt {len(optimization_log)}. History now contains {len(optimization_log)} entries."

    def get_optimization_history(self):
        """
        Returns the full list of all previous quantization attempts and their results.
        """
        if not optimization_log:
            return "No attempts have been recorded yet."
        return optimization_log

    def clear(self) -> None:
      """Clears all recorded optimization attempts."""
      self.optimization_log.clear()
