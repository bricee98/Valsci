"""Preflight and submission helpers for claim runs and arenas."""

from __future__ import annotations

import uuid
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional

from app.services.claim_store import (
    ClaimStore,
    candidate_color_for_index,
    candidate_prefix_for_index,
    normalize_claim_text,
)
from app.services.cost_estimator import CostEstimator
from app.services.provider_catalog import ProviderCatalog
from app.services.prompt_store import default_prompt_provenance
from app.services.stage_execution import (
    TASK_NAMES,
    next_stage,
    normalize_execution_settings,
    normalize_stop_after,
    skip_stages_for_execution,
    start_status_for_stage,
)


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

    @staticmethod
    def _start_stage_for_status(status: Optional[str]) -> str:
        mapping = {
            "queued": "query_generation",
            "ready_for_search": "paper_analysis",
            "ready_for_analysis": "venue_scoring",
        }
        return mapping.get(str(status or "").strip().lower(), "query_generation")

    @staticmethod
    def _claim_seed_for_continuation(claim_data: Dict[str, Any], next_stage_name: str, seed_from_run_id: str) -> Dict[str, Any]:
        seeded = deepcopy(claim_data or {})
        seeded["status"] = start_status_for_stage(next_stage_name)
        seeded["report"] = None
        seeded["stop_after"] = next_stage_name
        seeded["completed_stage"] = None
        seeded["is_stage_checkpoint"] = next_stage_name != "final_report"
        seeded["seed_from_run_id"] = seed_from_run_id
        return seeded

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
            candidate_index = candidate.get("candidate_index", index) if isinstance(candidate, dict) else index
            try:
                candidate_index = int(candidate_index)
            except Exception:
                candidate_index = index
            candidate_prefix = (
                candidate.get("candidate_prefix") if isinstance(candidate, dict) else None
            ) or candidate_prefix_for_index(candidate_index)
            candidate_color = (
                candidate.get("candidate_color") if isinstance(candidate, dict) else None
            ) or candidate_color_for_index(candidate_index)
            candidates.append(
                {
                    "candidate_id": (
                        candidate.get("candidate_id") if isinstance(candidate, dict) else None
                    ) or f"candidate-{candidate_index}",
                    "candidate_index": candidate_index,
                    "candidate_prefix": candidate_prefix,
                    "candidate_color": candidate_color,
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
        stop_after: str,
        start_stage: str = "query_generation",
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
                else:
                    skip_stages = skip_stages_for_execution(
                        stop_after=stop_after,
                        start_stage=start_stage,
                        reuse_query_generation=(
                            execution_mode == "reuse_retrieval" and len(candidates) > 1 and candidate_index > 0
                        ),
                    )

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
                        "stop_after": stop_after,
                        "start_stage": start_stage,
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
                    "stop_after": stop_after,
                    "start_stage": start_stage,
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
        stop_after: str = "final_report",
        start_stage: str = "query_generation",
    ) -> Dict[str, Any]:
        search_config = dict(search_config or self.default_search_config())
        duplicate_strategy = self._normalize_duplicate_strategy(duplicate_strategy)
        execution_mode, stop_after = normalize_execution_settings(execution_mode, stop_after)
        plan = self._claim_plan(claims, duplicate_strategy)
        estimates = self._candidate_estimates(
            unique_claims=plan["unique_claims"],
            candidates=candidates,
            search_config=search_config,
            execution_mode=execution_mode,
            stop_after=stop_after,
            start_stage=start_stage,
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
            "stop_after": stop_after,
            "start_stage": start_stage,
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
        stop_after: str = "final_report",
        cost_confirmation: Optional[Dict[str, Any]] = None,
        duplicate_strategy: str = "rerun",
        create_arena: bool = False,
        arena_title: Optional[str] = None,
        review_type: str = "regular",
        start_stage: str = "query_generation",
    ) -> Dict[str, Any]:
        duplicate_strategy = self._normalize_duplicate_strategy(duplicate_strategy)
        execution_mode, stop_after = normalize_execution_settings(execution_mode, stop_after)
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
            stop_after=stop_after,
            start_stage=start_stage,
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
                current_stage=stop_after,
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
                    stop_after=stop_after,
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
                    status=start_status_for_stage(start_stage),
                    source="arena" if create_arena else "standard",
                    candidate_id=candidate.get("candidate_id"),
                    candidate_index=candidate.get("candidate_index"),
                    candidate_prefix=candidate.get("candidate_prefix"),
                    candidate_label=candidate.get("label"),
                    candidate_color=candidate.get("candidate_color"),
                    prompt_provenance=default_prompt_provenance(),
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
            self.claim_store.append_arena_stage_history(
                arena_record,
                stage=stop_after,
                run_ids=[run["run_id"] for run in created_runs],
                source="initial_submit",
            )

        return {
            "batch_id": primary_batch_tag,
            "batch_tags": normalized_batch_tags,
            "arena_id": arena_record["arena_id"] if arena_record else None,
            "created_runs": [self.claim_store.build_run_summary(run) for run in created_runs],
            "reused_existing": reused_existing,
            "preflight": preflight,
            "execution_mode": execution_mode,
            "stop_after": stop_after,
        }

    def _normalize_continue_decisions(
        self,
        arena: Dict[str, Any],
        decisions: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        decision_map = {
            item.get("claim_key"): item
            for item in (decisions or [])
            if isinstance(item, dict) and isinstance(item.get("claim_key"), str)
        }
        normalized: List[Dict[str, Any]] = []
        for group in arena.get("claim_groups", []):
            claim_key = group.get("claim_key")
            runs = list(group.get("runs") or [])
            chosen = decision_map.get(claim_key, {})
            skip_claim = bool(chosen.get("skip_claim"))
            selected_run_id = chosen.get("selected_run_id")
            if not skip_claim and not selected_run_id and len(runs) == 1:
                selected_run_id = runs[0].get("run_id")
            normalized.append(
                {
                    "claim_key": claim_key,
                    "text": group.get("text", ""),
                    "skip_claim": skip_claim,
                    "selected_run_id": selected_run_id,
                    "available_run_ids": [run.get("run_id") for run in runs],
                }
            )
        return normalized

    def continue_arena_preflight(
        self,
        *,
        arena_id: str,
        decisions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        arena = self.claim_store.get_arena(arena_id)
        if not arena:
            raise ValueError("Arena not found.")
        current_stage = normalize_stop_after(arena.get("current_stage"))
        next_stage_name = next_stage(current_stage)
        if not next_stage_name:
            raise ValueError("Arena is already at the final stage.")

        normalized_decisions = self._normalize_continue_decisions(arena, decisions)
        claims_payload: List[Dict[str, Any]] = []
        total_expected_cost = 0.0
        total_upper_bound_cost = 0.0
        missing_pricing_models = set()
        pricing_complete = True

        for decision in normalized_decisions:
            action = "skip" if decision["skip_claim"] else "continue"
            claim_entry = {
                "claim_key": decision["claim_key"],
                "text": decision["text"],
                "action": action,
                "selected_run_id": decision["selected_run_id"],
            }
            available_runs = [
                self.claim_store.get_run(run_id)
                for run_id in decision.get("available_run_ids", [])
                if isinstance(run_id, str) and run_id.strip()
            ]
            available_runs = [run for run in available_runs if run]
            if decision["skip_claim"]:
                claims_payload.append(claim_entry)
                continue
            incomplete_runs = [
                run["run_id"]
                for run in available_runs
                if run.get("completed_stage") != current_stage
            ]
            if incomplete_runs:
                raise ValueError(
                    f"Claim {decision['claim_key']} still has candidates in progress at stage {current_stage}: "
                    + ", ".join(incomplete_runs)
                )
            selected_run_id = decision.get("selected_run_id")
            if not isinstance(selected_run_id, str) or not selected_run_id.strip():
                raise ValueError(f"A winner must be selected for claim {decision['claim_key']} or the claim must be skipped.")
            if selected_run_id not in set(decision["available_run_ids"]):
                raise ValueError(f"Selected run {selected_run_id} is not part of the current arena stage.")
            run_record = self.claim_store.get_run(selected_run_id)
            if not run_record:
                raise ValueError(f"Selected run not found: {selected_run_id}")
            if run_record.get("arena_id") != arena_id:
                raise ValueError(f"Selected run {selected_run_id} does not belong to arena {arena_id}.")
            if run_record.get("completed_stage") != current_stage:
                raise ValueError(f"Selected run {selected_run_id} is not completed at stage {current_stage}.")
            claim_data = self.claim_store.load_claim_data_for_run(run_record) or run_record.get("claim_data") or {}
            if not isinstance(claim_data, dict) or not claim_data:
                raise ValueError(f"Saved stage output is unavailable for run {selected_run_id}.")
            estimate = CostEstimator(
                run_record.get("provider_snapshot") or {},
                run_record.get("model_overrides") or {},
            ).estimate_run(
                run_record.get("text", ""),
                run_record.get("search_config") or self.default_search_config(),
                skip_stages=skip_stages_for_execution(
                    stop_after=next_stage_name,
                    start_stage=next_stage_name,
                ),
            )
            total_expected_cost = round(total_expected_cost + estimate["expected"]["cost_usd"], 10)
            total_upper_bound_cost = round(total_upper_bound_cost + estimate["upper_bound"]["cost_usd"], 10)
            missing_pricing_models.update(estimate.get("missing_pricing_models", []))
            pricing_complete = pricing_complete and bool(estimate.get("pricing_complete", True))
            claim_entry["estimate"] = estimate
            claim_entry["source_run"] = self.claim_store.build_run_summary(run_record)
            claim_entry["claim_data_present"] = isinstance(claim_data, dict) and bool(claim_data)
            claims_payload.append(claim_entry)

        return {
            "arena_id": arena_id,
            "current_stage": current_stage,
            "next_stage": next_stage_name,
            "claims": claims_payload,
            "totals": {
                "claim_count": len(claims_payload),
                "run_count": sum(1 for item in claims_payload if item["action"] == "continue"),
                "expected_cost_usd": total_expected_cost,
                "upper_bound_cost_usd": total_upper_bound_cost,
                "pricing_complete": pricing_complete,
                "missing_pricing_models": sorted(missing_pricing_models),
            },
            "execution_mode": arena.get("execution_mode", "full_pipeline"),
        }

    def continue_arena(
        self,
        *,
        arena_id: str,
        decisions: Optional[List[Dict[str, Any]]] = None,
        cost_confirmation: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not (cost_confirmation or {}).get("accepted"):
            raise ValueError("Cost confirmation is required before continuation.")
        preflight = self.continue_arena_preflight(arena_id=arena_id, decisions=decisions)
        if not preflight["totals"]["pricing_complete"]:
            raise ValueError(
                "Missing pricing metadata for one or more remote models: "
                + ", ".join(preflight["totals"]["missing_pricing_models"])
            )

        arena = self.claim_store.get_arena(arena_id)
        if not arena:
            raise ValueError("Arena not found.")

        created_runs = []
        continue_decisions = []
        for claim_entry in preflight["claims"]:
            continue_decisions.append(
                {
                    "claim_key": claim_entry["claim_key"],
                    "selected_run_id": claim_entry.get("selected_run_id"),
                    "action": claim_entry["action"],
                }
            )
            if claim_entry["action"] != "continue":
                continue
            selected_run_id = claim_entry["selected_run_id"]
            source_run = self.claim_store.get_run(selected_run_id)
            if not source_run:
                raise ValueError(f"Selected run not found: {selected_run_id}")
            claim_record = self.claim_store.get_claim(source_run["claim_key"])
            if not claim_record:
                raise ValueError(f"Claim not found for selected run: {selected_run_id}")
            claim_data = self.claim_store.load_claim_data_for_run(source_run) or source_run.get("claim_data") or {}
            if not isinstance(claim_data, dict) or not claim_data:
                raise ValueError(f"Saved stage output is unavailable for run {selected_run_id}.")
            seeded_claim_data = self._claim_seed_for_continuation(
                claim_data,
                preflight["next_stage"],
                selected_run_id,
            )
            new_run = self.claim_store.create_run(
                claim_record=claim_record,
                batch_tags=arena.get("batch_tags", []),
                arena_id=arena_id,
                execution_mode=arena.get("execution_mode", "full_pipeline"),
                stop_after=preflight["next_stage"],
                provider_snapshot=source_run.get("provider_snapshot") or {},
                model_overrides=source_run.get("model_overrides") or {},
                search_config=source_run.get("search_config") or self.default_search_config(),
                bibliometric_config=source_run.get("bibliometric_config") or self.default_bibliometric_config(),
                cost_estimate=claim_entry.get("estimate"),
                cost_confirmation=cost_confirmation,
                transport_batch_id=arena_id,
                review_type=source_run.get("review_type", "regular"),
                status=start_status_for_stage(preflight["next_stage"]),
                source="arena_continue",
                seed_from_run_id=selected_run_id,
                candidate_id=source_run.get("candidate_id"),
                candidate_index=source_run.get("candidate_index"),
                candidate_prefix=source_run.get("candidate_prefix"),
                candidate_label=source_run.get("candidate_label"),
                candidate_color=source_run.get("candidate_color"),
                prompt_provenance=source_run.get("prompt_provenance") or default_prompt_provenance(),
            )
            self.claim_store.materialize_run_to_queue(
                new_run,
                status=start_status_for_stage(preflight["next_stage"]),
                seeded_claim_data=seeded_claim_data,
            )
            created_runs.append(new_run)

        updated_arena = self.claim_store.append_arena_stage_history(
            arena,
            stage=preflight["next_stage"],
            run_ids=[run["run_id"] for run in created_runs],
            continue_decisions=continue_decisions,
            source="continue",
        )
        return {
            "arena_id": arena_id,
            "batch_id": arena_id,
            "created_runs": [self.claim_store.build_run_summary(run) for run in created_runs],
            "preflight": preflight,
            "execution_mode": updated_arena.get("execution_mode", "full_pipeline"),
            "stop_after": preflight["next_stage"],
            "skipped_claims": [item["claim_key"] for item in preflight["claims"] if item["action"] == "skip"],
        }

    @staticmethod
    def _candidate_public_summary(candidate: Dict[str, Any]) -> Dict[str, Any]:
        snapshot = candidate["provider_snapshot"]
        return {
            "candidate_id": candidate.get("candidate_id"),
            "candidate_index": candidate.get("candidate_index"),
            "candidate_prefix": candidate.get("candidate_prefix"),
            "candidate_color": candidate.get("candidate_color"),
            "provider_id": candidate.get("provider_id"),
            "label": candidate.get("label"),
            "provider_type": snapshot.get("provider_type"),
            "default_model": snapshot.get("default_model"),
            "model_overrides": candidate.get("model_overrides", {}),
        }
