import asyncio
import logging
import os
from agents import Agent, Runner

from grader.agentTools import QuantizationRecommendation, outputThinking
from grader import prompts
from shared_types import RecordValue, RecommendationRecord
logger = logging.getLogger(__name__)


class GraderService:
    """Wrapper around the grader agent for synchronous and async execution."""

    def __init__(
        self,
        model: str = "gpt-4o",
        output_type=QuantizationRecommendation,
        validation_retries: int | None = None,
    ) -> None:
        self.agent = Agent(
            name="Grader",
            instructions=prompts.build_quantization_instructions,
            model=model,
            output_type=output_type,
            tools=[outputThinking]
        )
        env_retries = os.getenv("GRADER_VALIDATION_RETRIES", "3")
        configured_retries = validation_retries if validation_retries is not None else self._to_int(env_retries, 3)
        self.max_validation_retries = max(0, configured_retries)
        self.max_tokens_upper_limit = max(1, self._to_int(os.getenv("GRADER_MAX_TOKENS_UPPER", "2048"), 2048))

    @staticmethod
    def _to_int(value: object, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_float(value: object, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _fallback_model_name(optimization_log: list, allowed_models: set[str]) -> str:
        if optimization_log:
            for entry in reversed(optimization_log):
                if isinstance(entry, dict):
                    model = str(entry.get("model name", "")).strip()
                    if model and (not allowed_models or model in allowed_models):
                        return model
        if allowed_models:
            return next(iter(allowed_models))
        return ""

    def _normalize_record(
        self,
        record: dict,
        optimization_log: list,
        model_names: list,
    ) -> RecommendationRecord:
        allowed_models = set(str(m) for m in model_names if str(m).strip())
        suggested_model = str(record.get("model name", "")).strip()
        if not suggested_model or (allowed_models and suggested_model not in allowed_models):
            suggested_model = self._fallback_model_name(optimization_log, allowed_models)

        max_tokens = self._to_int(record.get("max tokens"), 512)
        max_tokens = min(self.max_tokens_upper_limit, max(1, max_tokens))
        temperature = max(0.0, self._to_float(record.get("temperature"), 0.2))
        top_p = self._to_float(record.get("top p"), 0.95)
        top_p = min(1.0, max(0.0, top_p))
        top_k = max(0, self._to_int(record.get("top k"), 40))
        repeat_penalty = max(0.0, self._to_float(record.get("repeat penalty"), 1.1))
        quality_score = self._to_int(record.get("quality score"), 1)
        quality_score = min(100, max(1, quality_score))

        return {
            "model name": suggested_model,
            "max tokens": max_tokens,
            "temperature": temperature,
            "top p": top_p,
            "top k": top_k,
            "repeat penalty": repeat_penalty,
            "quality score": quality_score,
        }

    def _is_valid_raw_record(self, record: object, model_names: list) -> bool:
        if not isinstance(record, dict):
            return False

        required_keys = {
            "model name",
            "max tokens",
            "temperature",
            "top p",
            "top k",
            "repeat penalty",
            "quality score",
        }
        if not required_keys.issubset(record.keys()):
            return False

        model_name = str(record.get("model name", "")).strip()
        if not model_name:
            return False

        allowed_models = set(str(m) for m in model_names if str(m).strip())
        if allowed_models and model_name not in allowed_models:
            return False

        max_tokens = self._to_int(record.get("max tokens"), -1)
        temperature = self._to_float(record.get("temperature"), -1.0)
        top_p = self._to_float(record.get("top p"), -1.0)
        top_k = self._to_int(record.get("top k"), -1)
        repeat_penalty = self._to_float(record.get("repeat penalty"), -1.0)
        quality_score = self._to_int(record.get("quality score"), -1)

        return (
            1 <= max_tokens <= self.max_tokens_upper_limit
            and temperature >= 0.0
            and 0.0 <= top_p <= 1.0
            and top_k >= 0
            and repeat_penalty >= 0.0
            and 1 <= quality_score <= 100
        )

    def run(
        self,
        optimization_log: list | None = None,
        last_inference: list | None = None,
        model_names: list | None = None,
    ) -> RecommendationRecord:
        if optimization_log is None:
            optimization_log = []
        if last_inference is None:
            last_inference = []
        if model_names is None:
            model_names = []

        async def _run() -> RecommendationRecord:
            result = await Runner.run(
                self.agent,
                "Analyze the information given and make your educated guess",
                context={
                    "optimization_log": optimization_log,
                    "last_inference": last_inference,
                    "model_names": model_names,
                },
            )
            output = result.final_output
            if hasattr(output, "model_dump"):
                return output.model_dump(by_alias=True)
            return output

        last_raw_record: RecommendationRecord | object = {}
        for attempt in range(self.max_validation_retries + 1):
            last_raw_record = asyncio.run(_run())
            if self._is_valid_raw_record(last_raw_record, model_names):
                return self._normalize_record(last_raw_record, optimization_log, model_names)
            logger.info(
                "Grader returned invalid recommendation on attempt %s/%s: %s",
                attempt + 1,
                self.max_validation_retries + 1,
                last_raw_record,
            )

        logger.info("Using normalized fallback after grader validation retries were exhausted.")
        if not isinstance(last_raw_record, dict):
            last_raw_record = {}
        return self._normalize_record(last_raw_record, optimization_log, model_names)
