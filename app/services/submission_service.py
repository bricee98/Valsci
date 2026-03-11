"""Preflight and submission helpers for claim runs and arenas."""

from __future__ import annotations

import uuid
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional

from app.services.claim_store import ClaimStore, normalize_claim_text
from app.services.cost_estimator import CostEstimator
from app.services.provider_catalog import ProviderCatalog


TASK_NAMES = ["query_generation", "paper_analysis", "venue_scoring", "final_report"]


def clean_model_overrides(payload: Optional[Dict[str, Any]]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for task in TASK_NAMES:
        value = None
        if isinstance(payload, dict):
            value = payload.get(task)
        if isinstance(value, str) and value.strip():
            overrides[task] = value.strip()
    return overrides


class SubmissionService:
    def __init__(self, claim_store: ClaimStore, provider_catalog: ProviderCatalog):
        self.claim_store = claim_store
        self.provider_catalog = provider_catalog

    def default_search_config(self) -> Dict[str, Any]:
        return {"num_queries": 5, "results_per_query": 5}

    def default_bibliometric_config(self) -> Dict[str, Any]:
        return {
            "use_bibliometrics": True,
            "author_impact_weight": 0.4,
            "citation_impact_weight": 0.4,
            "venue_impact_weight": 0.2,
        }

    def resolve_candidates(self, raw_candidates: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        providers = self.provider_catalog.list_providers()
        default_provider = providers[0] if providers else None
        if not raw_candidates:
            if not default_provider:
                raise ValueError("No providers are configured.")
            raw_candidates = [{"provider_id": default_provider["provider_id"], "model_overrides": {}}]

        candidates: List[Dict[str, Any]] = []
        for index, candidate in enumerate(raw_candidates):
            provider_id = candidate.get("provider_id") if isinstance(candidate, dict) else None
            if not provider_id:
                raise ValueError("Each candidate must include provider_id.")
            provider_snapshot = self.provider_catalog.build_snapshot(provider_id)
            model_overrides = clean_model_overrides(candidate.get("model_overrides") if isinstance(candidate, dict) else {})
            label = candidate.get("label") if isinstance(candidate, dict) else None
            candidates.append(
                {
                    "candidate_id": f"candidate-{index}",
                    "provider_id": provider_id,
                    "label": label or provider_snapshot.get("label") or provider_id,
                    "provider_snapshot": provider_snapshot,
                    "model_overrides": model_overrides,
                }
            )
        return candidates

    @staticmethod
    def _normalize_duplicate_strategy(value: Optional[str]) -> str:
        strategy = str(value or "rerun").strip().lower()
        if strategy not in {"rerun", "view"}:
            raise ValueError("duplicate_strategy must be either 'rerun' or 'view'.")
        return strategy

    @staticmethod
    def _normalize_execution_mode(value: Optional[str]) -> str:
        mode = str(value or "full_pipeline").strip().lower()
        if mode not in {"full_pipeline", "reuse_retrieval"}:
            raise ValueError("execution_mode must be either 'full_pipeline' or 'reuse_retrieval'.")
        return mode

    def _claim_plan(self, claims: List[str], duplicate_strategy: str) -> Dict[str, Any]:
        normalized_groups: Dict[str, List[int]] = defaultdict(list)
        claim_entries: List[Dict[str, Any]] = []
        unique_claims: List[Dict[str, Any]] = []
        unique_by_normalized: Dict[str, Dict[str, Any]] = {}

        for index, claim_text in enumerate(claims):
            normalized = normalize_claim_text(claim_text)
            normalized_groups[normalized].append(index)

            unique_entry = unique_by_normalized.get(normalized)
            if unique_entry is None:
                existing_claim = self.claim_store.find_claim_by_text(claim_text)
                existing_runs = (
                    self.claim_store.list_runs_for_claim(existing_claim["claim_key"])
                    if existing_claim
                    else []
                )
                unique_entry = {
                    "text": claim_text,
                    "normalized_text": normalized,
                    "claim_key": existing_claim["claim_key"] if existing_claim else None,
                    "existing_claim": existing_claim is not None,
                    "existing_run_count": len(existing_runs),
                    "existing_latest_run_id": existing_claim.get("latest_run_id") if existing_claim else None,
                    "source_indices": [],
                }
                unique_by_normalized[normalized] = unique_entry
                unique_claims.append(unique_entry)

            unique_entry["source_indices"].append(index)
            claim_entries.append(
                {
                    "text": claim_text,
                    "normalized_text": normalized,
                    "claim_key": unique_entry["claim_key"],
                    "existing_claim": unique_entry["existing_claim"],
                    "existing_run_count": unique_entry["existing_run_count"],
                    "existing_latest_run_id": unique_entry["existing_latest_run_id"],
                }
            )

        duplicates = [
            {"normalized_text": normalized, "indices": indices}
            for normalized, indices in normalized_groups.items()
            if len(indices) > 1
        ]

        for entry in claim_entries:
            indices = normalized_groups[entry["normalized_text"]]
            entry["duplicate_group_indices"] = list(indices)
            entry["duplicate_group_size"] = len(indices)
            entry["duplicate_input"] = len(indices) > 1

        for unique_entry in unique_claims:
            should_reuse_existing = (
                duplicate_strategy == "view"
                and unique_entry.get("existing_claim")
                and unique_entry.get("existing_latest_run_id")
            )
            unique_entry["duplicate_input_count"] = len(unique_entry["source_indices"])
            unique_entry["submission_action"] = "view_existing" if should_reuse_existing else "create_run"
            unique_entry["will_create_run"] = not should_reuse_existing

        return {
            "claims": claim_entries,
            "duplicates": duplicates,
            "unique_claims": unique_claims,
        }

    def _candidate_estimates(
        self,
        *,
        unique_claims: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        search_config: Dict[str, Any],
        execution_mode: str,
    ) -> Dict[str, Any]:
        candidate_estimates = []
        total_expected_cost = 0.0
        total_upper_bound_cost = 0.0
        total_run_count = 0
        overall_pricing_complete = True
        missing_pricing_models = set()

        for candidate_index, candidate in enumerate(candidates):
            estimator = CostEstimator(candidate["provider_snapshot"], candidate.get("model_overrides"))
            run_estimates = []
            expected_cost = 0.0
            upper_bound_cost = 0.0
            candidate_pricing_complete = True
            candidate_missing_pricing = set()
            candidate_run_count = 0

            for unique_claim in unique_claims:
                action = unique_claim["submission_action"]
                skip_stages = set()
                if action != "create_run":
                    skip_stages = set(TASK_NAMES)
                elif execution_mode == "reuse_retrieval" and len(candidates) > 1 and candidate_index > 0:
                    skip_stages = {"query_generation"}

                estimate = estimator.estimate_run(
                    unique_claim["text"],
                    search_config,
                    skip_stages=skip_stages,
                )
                if action == "create_run":
                    candidate_run_count += 1
                    expected_cost = round(expected_cost + estimate["expected"]["cost_usd"], 10)
                    upper_bound_cost = round(upper_bound_cost + estimate["upper_bound"]["cost_usd"], 10)
                candidate_pricing_complete = candidate_pricing_complete and bool(estimate.get("pricing_complete", True))
                candidate_missing_pricing.update(estimate.get("missing_pricing_models", []))
                run_estimates.append(
                    {
                        "text": unique_claim["text"],
                        "normalized_text": unique_claim["normalized_text"],
                        "claim_key": unique_claim.get("claim_key"),
                        "source_indices": list(unique_claim["source_indices"]),
                        "duplicate_input_count": unique_claim["duplicate_input_count"],
                        "existing_claim": bool(unique_claim.get("existing_claim")),
                        "action": action,
                        "estimate": estimate,
                    }
                )

            total_expected_cost = round(total_expected_cost + expected_cost, 10)
            total_upper_bound_cost = round(total_upper_bound_cost + upper_bound_cost, 10)
            total_run_count += candidate_run_count
            overall_pricing_complete = overall_pricing_complete and candidate_pricing_complete
            missing_pricing_models.update(candidate_missing_pricing)
            candidate_estimates.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "provider_id": candidate["provider_id"],
                    "label": candidate["label"],
                    "model_overrides": candidate.get("model_overrides", {}),
                    "default_model": candidate["provider_snapshot"].get("default_model"),
                    "expected_cost_usd": expected_cost,
                    "upper_bound_cost_usd": upper_bound_cost,
                    "pricing_complete": candidate_pricing_complete,
                    "missing_pricing_models": sorted(candidate_missing_pricing),
                    "run_count": candidate_run_count,
                    "runs": run_estimates,
                }
            )

        return {
            "candidates": candidate_estimates,
            "totals": {
                "run_count": total_run_count,
                "expected_cost_usd": total_expected_cost,
                "upper_bound_cost_usd": total_upper_bound_cost,
                "pricing_complete": overall_pricing_complete,
                "missing_pricing_models": sorted(missing_pricing_models),
            },
        }

    def preflight(
        self,
        *,
        claims: List[str],
        candidates: List[Dict[str, Any]],
        search_config: Optional[Dict[str, Any]] = None,
        duplicate_strategy: str = "rerun",
        execution_mode: str = "full_pipeline",
    ) -> Dict[str, Any]:
        search_config = dict(search_config or self.default_search_config())
        duplicate_strategy = self._normalize_duplicate_strategy(duplicate_strategy)
        execution_mode = self._normalize_execution_mode(execution_mode)
        plan = self._claim_plan(claims, duplicate_strategy)
        estimates = self._candidate_estimates(
            unique_claims=plan["unique_claims"],
            candidates=candidates,
            search_config=search_config,
            execution_mode=execution_mode,
        )

        return {
            "claims": plan["claims"],
            "duplicates": plan["duplicates"],
            "unique_claims": deepcopy(plan["unique_claims"]),
            "candidates": estimates["candidates"],
            "totals": {
                "claim_count": len(claims),
                "raw_claim_count": len(claims),
                "unique_claim_count": len(plan["unique_claims"]),
                "duplicate_input_count": max(0, len(claims) - len(plan["unique_claims"])),
                "candidate_count": len(candidates),
                "run_count": estimates["totals"]["run_count"],
                "reused_existing_count": sum(
                    1 for claim in plan["unique_claims"] if claim["submission_action"] == "view_existing"
                ),
                "expected_cost_usd": estimates["totals"]["expected_cost_usd"],
                "upper_bound_cost_usd": estimates["totals"]["upper_bound_cost_usd"],
                "pricing_complete": estimates["totals"]["pricing_complete"],
                "missing_pricing_models": estimates["totals"]["missing_pricing_models"],
            },
            "duplicate_strategy": duplicate_strategy,
            "execution_mode": execution_mode,
        }

    def submit(
        self,
        *,
        claims: List[str],
        candidates: List[Dict[str, Any]],
        search_config: Optional[Dict[str, Any]] = None,
        bibliometric_config: Optional[Dict[str, Any]] = None,
        batch_tags: Optional[Iterable[str]] = None,
        execution_mode: str = "full_pipeline",
        cost_confirmation: Optional[Dict[str, Any]] = None,
        duplicate_strategy: str = "rerun",
        create_arena: bool = False,
        arena_title: Optional[str] = None,
        review_type: str = "regular",
    ) -> Dict[str, Any]:
        duplicate_strategy = self._normalize_duplicate_strategy(duplicate_strategy)
        execution_mode = self._normalize_execution_mode(execution_mode)
        if not (cost_confirmation or {}).get("accepted"):
            raise ValueError("Cost confirmation is required before submission.")

        search_config = dict(search_config or self.default_search_config())
        bibliometric_config = dict(bibliometric_config or self.default_bibliometric_config())
        preflight = self.preflight(
            claims=claims,
            candidates=candidates,
            search_config=search_config,
            duplicate_strategy=duplicate_strategy,
            execution_mode=execution_mode,
        )
        if not preflight["totals"]["pricing_complete"]:
            raise ValueError(
                "Missing pricing metadata for one or more remote models: "
                + ", ".join(preflight["totals"]["missing_pricing_models"])
            )

        primary_batch_tag = next(iter(batch_tags or []), None) or uuid.uuid4().hex[:8]
        normalized_batch_tags = [primary_batch_tag, *[tag for tag in (batch_tags or []) if tag != primary_batch_tag]]

        arena_record = None
        if create_arena:
            arena_record = self.claim_store.create_arena(
                title=arena_title or f"Arena {primary_batch_tag}",
                batch_tags=normalized_batch_tags,
                execution_mode=execution_mode,
                candidates=[self._candidate_public_summary(candidate) for candidate in candidates],
            )
            normalized_batch_tags = [arena_record["arena_id"], *normalized_batch_tags]

        created_runs = []
        reused_existing = []
        claim_keys = []
        candidate_run_plans: Dict[tuple[str, str], Dict[str, Any]] = {}
        for candidate_plan in preflight["candidates"]:
            candidate_id = candidate_plan["candidate_id"]
            for run_plan in candidate_plan["runs"]:
                candidate_run_plans[(candidate_id, run_plan["normalized_text"])] = run_plan

        for claim_plan in preflight["unique_claims"]:
            claim_text = claim_plan["text"]
            claim_record, _ = self.claim_store.get_or_create_claim(
                claim_text,
                batch_tags=normalized_batch_tags,
                metadata={"submitted_from": "arena" if create_arena else "standard"},
            )
            claim_keys.append(claim_record["claim_key"])
            if claim_plan["submission_action"] == "view_existing" and claim_record.get("latest_run_id"):
                reused_existing.append(
                    {
                        "claim_key": claim_record["claim_key"],
                        "latest_run_id": claim_record.get("latest_run_id"),
                        "source_indices": list(claim_plan["source_indices"]),
                    }
                )
                continue

            baseline_run = None
            for candidate_index, candidate in enumerate(candidates):
                candidate_run_plan = candidate_run_plans.get((candidate["candidate_id"], claim_plan["normalized_text"]))
                if candidate_run_plan is None or candidate_run_plan["action"] != "create_run":
                    continue

                run_record = self.claim_store.create_run(
                    claim_record=claim_record,
                    batch_tags=normalized_batch_tags,
                    arena_id=arena_record["arena_id"] if arena_record else None,
                    execution_mode=execution_mode,
                    provider_snapshot={
                        **candidate["provider_snapshot"],
                        "label": candidate["label"],
                    },
                    model_overrides=candidate.get("model_overrides"),
                    search_config=search_config,
                    bibliometric_config=bibliometric_config,
                    cost_estimate=candidate_run_plan["estimate"],
                    cost_confirmation=cost_confirmation,
                    transport_batch_id=arena_record["arena_id"] if arena_record else primary_batch_tag,
                    review_type=review_type,
                    status="queued",
                    source="arena" if create_arena else "standard",
                )
                if execution_mode == "reuse_retrieval" and len(candidates) > 1:
                    if candidate_index == 0:
                        baseline_run = run_record
                    else:
                        run_record["status"] = "waiting_for_baseline"
                        run_record["reuse_from_run_id"] = baseline_run["run_id"] if baseline_run else None
                        self.claim_store.save_run(run_record)
                        created_runs.append(run_record)
                        continue

                self.claim_store.materialize_run_to_queue(run_record)
                created_runs.append(run_record)

        if arena_record:
            arena_record["claim_keys"] = claim_keys
            arena_record["run_ids"] = [run["run_id"] for run in created_runs]
            self.claim_store.save_arena(arena_record)

        return {
            "batch_id": primary_batch_tag,
            "batch_tags": normalized_batch_tags,
            "arena_id": arena_record["arena_id"] if arena_record else None,
            "created_runs": [self.claim_store.build_run_summary(run) for run in created_runs],
            "reused_existing": reused_existing,
            "preflight": preflight,
            "execution_mode": execution_mode,
        }

    @staticmethod
    def _candidate_public_summary(candidate: Dict[str, Any]) -> Dict[str, Any]:
        snapshot = candidate["provider_snapshot"]
        return {
            "candidate_id": candidate.get("candidate_id"),
            "provider_id": candidate.get("provider_id"),
            "label": candidate.get("label"),
            "provider_type": snapshot.get("provider_type"),
            "default_model": snapshot.get("default_model"),
            "model_overrides": candidate.get("model_overrides", {}),
        }
