from shared_types import RecommendationRecord

class LoggingService:
    optimization_log: list[RecommendationRecord]

    def __init__(self):
        self.optimization_log = []

    def record_attempt(self, record: RecommendationRecord):
        """
        Stores the result of test in the log.
        """

        self.optimization_log.append(record)
        return (
            f"Logged attempt {len(self.optimization_log)}. "
            f"History now contains {len(self.optimization_log)} entries."
        )

    def get_optimization_history(self) -> list[RecommendationRecord]:
        """
        Returns the full list of all previous attempts and their results.
        """
        if not self.optimization_log:
            return []
        return self.optimization_log

    def get_most_recent_recommendation(self) -> RecommendationRecord | None:
        if not self.optimization_log:
            return None
        return self.optimization_log[-1]

    def get_most_recent_suggestion(self) -> str:
        #gets just the string (name) of the most recent entry in the optimization log
        most_recent = self.get_most_recent_recommendation()
        if most_recent is None:
            return ""
        return str(most_recent.get("model name", ""))
        

    def clear(self) -> None:
      """Clears all recorded optimization attempts."""
      self.optimization_log.clear()
