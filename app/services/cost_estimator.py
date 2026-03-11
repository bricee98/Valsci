"""Submission-time cost estimation for claim runs and arenas."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set

from app.services.llm.gateway import DEFAULT_TASK_CONFIG, LLMTask
from app.services.llm.model_registry import DEFAULT_MODEL_INFO, ModelRegistry


LOCAL_PROVIDER_NAMES = {"local", "ollama", "llamacpp"}
RUN_STAGES = [
    LLMTask.QUERY_GENERATION,
    LLMTask.PAPER_ANALYSIS,
    LLMTask.VENUE_SCORING,
    LLMTask.FINAL_REPORT,
]


class CostEstimator:
    def __init__(self, provider_snapshot: Dict[str, Any], model_overrides: Optional[Dict[str, str]] = None):
        self.provider_snapshot = provider_snapshot or {}
        registry_overrides = dict(self.provider_snapshot.get("model_registry_overrides") or {})
        for model_name, model_info in DEFAULT_MODEL_INFO.items():
            registry_overrides.setdefault(
                model_name,
                {
                    "context_window_tokens": model_info.context_window_tokens,
                    "max_output_tokens_default": model_info.max_output_tokens_default,
                    "supports_temperature": model_info.supports_temperature,
                    "supports_json_schema_or_json_mode": model_info.supports_json_mode,
                    "input_cost_per_million": model_info.input_cost_per_million,
                    "output_cost_per_million": model_info.output_cost_per_million,
                },
            )
        self.model_registry = ModelRegistry(model_overrides=registry_overrides)
        self.known_models = set(registry_overrides.keys())
        self.model_overrides = model_overrides or {}

    def estimate_run(
        self,
        claim_text: str,
        search_config: Dict[str, Any],
        *,
        skip_stages: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        num_queries = int(search_config.get("num_queries", 5) or 5)
        results_per_query = int(search_config.get("results_per_query", 5) or 5)
        candidate_papers = max(1, num_queries * results_per_query)
        expected_relevant_papers = max(1, round(candidate_papers * 0.4))
        claim_len = max(1, len(claim_text or ""))
        skipped = set(skip_stages or [])

        expected_stage_tokens = {
            LLMTask.QUERY_GENERATION: {
                "input_tokens": int(300 + claim_len / 4),
                "output_tokens": int(min(220, 40 * num_queries)),
            },
            LLMTask.PAPER_ANALYSIS: {
                "input_tokens": int(candidate_papers * (1400 + claim_len / 8)),
                "output_tokens": int(candidate_papers * 320),
            },
            LLMTask.VENUE_SCORING: {
                "input_tokens": int(expected_relevant_papers * 260),
                "output_tokens": int(expected_relevant_papers * 40),
            },
            LLMTask.FINAL_REPORT: {
                "input_tokens": int(1600 + expected_relevant_papers * 360),
                "output_tokens": int(900),
            },
        }

        upper_stage_tokens = {
            LLMTask.QUERY_GENERATION: {
                "input_tokens": int(500 + claim_len / 3),
                "output_tokens": self._max_output_tokens(LLMTask.QUERY_GENERATION),
            },
            LLMTask.PAPER_ANALYSIS: {
                "input_tokens": int(candidate_papers * 4200),
                "output_tokens": int(candidate_papers * self._max_output_tokens(LLMTask.PAPER_ANALYSIS)),
            },
            LLMTask.VENUE_SCORING: {
                "input_tokens": int(candidate_papers * 400),
                "output_tokens": int(candidate_papers * self._max_output_tokens(LLMTask.VENUE_SCORING)),
            },
            LLMTask.FINAL_REPORT: {
                "input_tokens": int(6000 + candidate_papers * 200),
                "output_tokens": self._max_output_tokens(LLMTask.FINAL_REPORT),
            },
        }

        expected = self._cost_breakdown(expected_stage_tokens, skip_stages=skipped)
        upper_bound = self._cost_breakdown(upper_stage_tokens, skip_stages=skipped)
        missing_pricing_models = sorted(
            set(expected.get("missing_pricing_models", [])) | set(upper_bound.get("missing_pricing_models", []))
        )
        return {
            "expected": expected,
            "upper_bound": upper_bound,
            "assumptions": {
                "num_queries": num_queries,
                "results_per_query": results_per_query,
                "candidate_papers": candidate_papers,
                "expected_relevant_papers": expected_relevant_papers,
            },
            "skipped_stages": sorted(skipped),
            "pricing_complete": not missing_pricing_models,
            "missing_pricing_models": missing_pricing_models,
        }

    def _cost_breakdown(
        self,
        stage_tokens: Dict[str, Dict[str, int]],
        *,
        skip_stages: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        skipped = skip_stages or set()
        breakdown: Dict[str, Any] = {
            "stages": {},
            "total_tokens": 0,
            "cost_usd": 0.0,
            "pricing_complete": True,
            "missing_pricing_models": [],
        }
        for stage, stats in stage_tokens.items():
            model_name = self._model_for_stage(stage)
            is_skipped = stage in skipped
            input_tokens = 0 if is_skipped else int(stats.get("input_tokens", 0))
            output_tokens = 0 if is_skipped else int(stats.get("output_tokens", 0))
            total_tokens = input_tokens + output_tokens
            pricing_defined = self._pricing_defined(model_name)
            if not pricing_defined:
                breakdown["pricing_complete"] = False
                breakdown["missing_pricing_models"].append(model_name)
            cost_usd = self._calculate_cost(model_name, input_tokens, output_tokens) if pricing_defined else 0.0
            breakdown["stages"][stage] = {
                "model": model_name,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "cost_usd": cost_usd,
                "skipped": is_skipped,
                "pricing_defined": pricing_defined,
            }
            breakdown["total_tokens"] += total_tokens
            breakdown["cost_usd"] = round(breakdown["cost_usd"] + cost_usd, 10)
        breakdown["missing_pricing_models"] = sorted(set(breakdown["missing_pricing_models"]))
        return breakdown

    def _calculate_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        provider_type = str(self.provider_snapshot.get("provider_type", "")).lower()
        cost = self.model_registry.calculate_cost(model_name, input_tokens, output_tokens)
        if provider_type in LOCAL_PROVIDER_NAMES and cost == 0.0:
            return 0.0
        return cost

    def _pricing_defined(self, model_name: str) -> bool:
        provider_type = str(self.provider_snapshot.get("provider_type", "")).lower()
        if provider_type in LOCAL_PROVIDER_NAMES:
            return True
        return model_name in self.known_models

    def _model_for_stage(self, stage: str) -> str:
        task_defaults = self.provider_snapshot.get("task_defaults") or {}
        return (
            self.model_overrides.get(stage)
            or task_defaults.get(stage)
            or task_defaults.get("default")
            or self.provider_snapshot.get("default_model")
            or "unknown-model"
        )

    @staticmethod
    def _max_output_tokens(stage: str) -> int:
        task_config = DEFAULT_TASK_CONFIG.get(stage, DEFAULT_TASK_CONFIG[LLMTask.GENERIC])
        return int(task_config.get("max_output_tokens", 1024))


def estimate_submission_costs(
    *,
    claims: List[str],
    provider_snapshot: Dict[str, Any],
    model_overrides: Optional[Dict[str, str]],
    search_config: Dict[str, Any],
    multiplier: int = 1,
) -> Dict[str, Any]:
    estimator = CostEstimator(provider_snapshot, model_overrides=model_overrides)
    claim_estimates = []
    totals = {
        "expected_cost_usd": 0.0,
        "upper_bound_cost_usd": 0.0,
        "pricing_complete": True,
        "missing_pricing_models": [],
    }
    for claim_text in claims:
        run_estimate = estimator.estimate_run(claim_text, search_config)
        totals["expected_cost_usd"] = round(
            totals["expected_cost_usd"] + run_estimate["expected"]["cost_usd"] * multiplier,
            10,
        )
        totals["upper_bound_cost_usd"] = round(
            totals["upper_bound_cost_usd"] + run_estimate["upper_bound"]["cost_usd"] * multiplier,
            10,
        )
        totals["pricing_complete"] = totals["pricing_complete"] and bool(run_estimate.get("pricing_complete", True))
        totals["missing_pricing_models"] = sorted(
            set(totals["missing_pricing_models"]) | set(run_estimate.get("missing_pricing_models", []))
        )
        claim_estimates.append({"text": claim_text, "estimate": run_estimate})
    return {
        "claims": claim_estimates,
        "multiplier": multiplier,
        "totals": totals,
    }
