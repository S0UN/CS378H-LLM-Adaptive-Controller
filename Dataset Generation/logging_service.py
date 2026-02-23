
from typing import Dict, List, Union

class LoggingService:
    optimization_log: List[Dict[str, Union[str, float]]]

    def __init__(self):
        self.optimization_log = []

    def record_attempt(self, record: Dict[str, Union[str, float]]):
        """
        Stores the result of a quantization test in the log.
        """

        self.optimization_log.append(record)
        return (
            f"Logged attempt {len(self.optimization_log)}. "
            f"History now contains {len(self.optimization_log)} entries."
        )

    def get_optimization_history(self) ->  List[Dict[str, Union[str, float]]]:
        """
        Returns the full list of all previous quantization attempts and their results.
        """
        if not self.optimization_log:
            return []
        return self.optimization_log

    def get_most_recent_suggestion(self) -> str:
        #gets just the string (name) of the most recent entry in the optimization log
        if not self.optimization_log:
            return ""
        most_recent = self.optimization_log[-1]
        return str(most_recent.get("model name", ""))
        

    def clear(self) -> None:
      """Clears all recorded optimization attempts."""
      self.optimization_log.clear()
