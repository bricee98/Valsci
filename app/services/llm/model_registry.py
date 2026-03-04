"""Model metadata registry used for routing, preflight, and costing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ModelInfo:
    context_window_tokens: int
    max_output_tokens_default: int
    supports_temperature: bool = True
    supports_json_mode: bool = True
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0


DEFAULT_MODEL_INFO: Dict[str, ModelInfo] = {
    "gpt-4o": ModelInfo(
        context_window_tokens=128000,
        max_output_tokens_default=4096,
        supports_temperature=True,
        supports_json_mode=True,
        input_cost_per_million=2.50,
        output_cost_per_million=10.00,
    ),
    "gpt-4o-mini": ModelInfo(
        context_window_tokens=128000,
        max_output_tokens_default=4096,
        supports_temperature=True,
        supports_json_mode=True,
        input_cost_per_million=0.15,
        output_cost_per_million=0.60,
    ),
    "gpt-5": ModelInfo(
        context_window_tokens=272000,
        max_output_tokens_default=8192,
        supports_temperature=False,
        supports_json_mode=True,
        input_cost_per_million=1.25,
        output_cost_per_million=10.00,
    ),
    "gpt-5-mini": ModelInfo(
        context_window_tokens=272000,
        max_output_tokens_default=8192,
        supports_temperature=False,
        supports_json_mode=True,
        input_cost_per_million=0.25,
        output_cost_per_million=2.00,
    ),
    "o3": ModelInfo(
        context_window_tokens=200000,
        max_output_tokens_default=4096,
        supports_temperature=False,
        supports_json_mode=True,
        input_cost_per_million=2.00,
        output_cost_per_million=8.00,
    ),
}


class ModelRegistry:
    def __init__(
        self,
        model_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        local_context_override: Optional[int] = None,
    ):
        self.models: Dict[str, ModelInfo] = dict(DEFAULT_MODEL_INFO)
        self.local_context_override = local_context_override
        if model_overrides:
            self.apply_overrides(model_overrides)

    def apply_overrides(self, overrides: Dict[str, Dict[str, Any]]) -> None:
        for model_name, values in overrides.items():
            existing = self.models.get(model_name, ModelInfo(8192, 1024, True, True, 0.0, 0.0))
            self.models[model_name] = ModelInfo(
                context_window_tokens=int(values.get("context_window_tokens", existing.context_window_tokens)),
                max_output_tokens_default=int(values.get("max_output_tokens_default", existing.max_output_tokens_default)),
                supports_temperature=bool(values.get("supports_temperature", existing.supports_temperature)),
                supports_json_mode=bool(
                    values.get(
                        "supports_json_schema_or_json_mode",
                        values.get("supports_json_mode", existing.supports_json_mode),
                    )
                ),
                input_cost_per_million=float(values.get("input_cost_per_million", existing.input_cost_per_million)),
                output_cost_per_million=float(values.get("output_cost_per_million", existing.output_cost_per_million)),
            )

    def set_runtime_context_window(self, model_name: str, context_window_tokens: int) -> None:
        existing = self.get_model_info(model_name)
        self.models[model_name] = ModelInfo(
            context_window_tokens=int(context_window_tokens),
            max_output_tokens_default=existing.max_output_tokens_default,
            supports_temperature=existing.supports_temperature,
            supports_json_mode=existing.supports_json_mode,
            input_cost_per_million=existing.input_cost_per_million,
            output_cost_per_million=existing.output_cost_per_million,
        )

    def get_model_info(self, model_name: str) -> ModelInfo:
        if model_name in self.models:
            return self.models[model_name]
        # Unknown models default to local override or conservative generic limits.
        default_context = self.local_context_override if self.local_context_override else 8192
        return ModelInfo(
            context_window_tokens=default_context,
            max_output_tokens_default=1024,
            supports_temperature=True,
            supports_json_mode=False,
            input_cost_per_million=0.0,
            output_cost_per_million=0.0,
        )

    def supports_temperature(self, model_name: str) -> bool:
        return self.get_model_info(model_name).supports_temperature

    def supports_json_mode(self, model_name: str) -> bool:
        return self.get_model_info(model_name).supports_json_mode

    def context_window(self, model_name: str) -> int:
        return self.get_model_info(model_name).context_window_tokens

    def default_max_output_tokens(self, model_name: str) -> int:
        return self.get_model_info(model_name).max_output_tokens_default

    def calculate_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        info = self.get_model_info(model_name)
        return round(
            (input_tokens * info.input_cost_per_million / 1_000_000.0)
            + (output_tokens * info.output_cost_per_million / 1_000_000.0),
            10,
        )

