from typing import Any

RecordValue = str | int | float
RecommendationRecord = dict[str, RecordValue]
TurnResult = dict[str, Any]
RowResult = list[TurnResult]
GenerationConfig = dict[str, int | float]
