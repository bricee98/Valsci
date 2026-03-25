"""Canonical claim/run store layered on top of the file-backed processor."""

from __future__ import annotations

import hashlib
import gzip
import json
import os
import re
import shutil
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.config.settings import Config
from app.services.llm.types import empty_usage
from app.services.prompt_store import default_prompt_provenance, sha256_text, stage_prompt_provenance
from app.services.stage_execution import (
    checkpoint_complete,
    continue_stage_for_run,
    normalize_stop_after,
    review_value_for_stage,
    stage_label,
    is_stage_checkpoint as stage_is_checkpoint,
)


WHITESPACE_RE = re.compile(r"\s+")
CANDIDATE_COLOR_PALETTE = [
    "#0f766e",
    "#c2410c",
    "#1d4ed8",
    "#b45309",
    "#be123c",
    "#4f46e5",
    "#0f766e",
]
RUN_STATUS_TO_STAGE = {
    "queued": "query_generation",
    "waiting_for_baseline": "query_generation",
    "ready_for_search": "paper_analysis",
    "ready_for_analysis": "paper_analysis",
    "processed": "final_report",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def collapse_whitespace(value: str) -> str:
    return WHITESPACE_RE.sub(" ", value or "").strip()


def normalize_claim_text(value: str) -> str:
    return collapse_whitespace(value).lower()


def claim_key_for_text(value: str) -> str:
    normalized = normalize_claim_text(value)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    open_fn = gzip.open if path.suffix == ".gz" else open
    records: List[Dict[str, Any]] = []
    with open_fn(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    os.replace(temp_path, path)


def _dedupe_str_list(values: Iterable[str]) -> List[str]:
    items: List[str] = []
    seen = set()
    for value in values:
        if not isinstance(value, str):
            continue
        text = value.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        items.append(text)
    return items


def candidate_prefix_for_index(index: int) -> str:
    if index < 26:
        return chr(ord("A") + index)
    return f"A{index + 1}"


def candidate_color_for_index(index: int) -> str:
    return CANDIDATE_COLOR_PALETTE[index % len(CANDIDATE_COLOR_PALETTE)]


class ClaimStore:
    def __init__(
        self,
        *,
        state_dir: Optional[str] = None,
        saved_jobs_dir: Optional[str] = None,
        queued_jobs_dir: Optional[str] = None,
        trace_dir: Optional[str] = None,
    ):
        self.state_dir = Path(state_dir or Config.STATE_DIR)
        self.saved_jobs_dir = Path(saved_jobs_dir or Config.SAVED_JOBS_DIR)
        self.queued_jobs_dir = Path(queued_jobs_dir or Config.QUEUED_JOBS_DIR)
        self.trace_dir = Path(trace_dir or Config.TRACE_DIR)
        self.archive_dir = self.state_dir / "migrations" / "archive"
        self.claims_dir = self.state_dir / "claims"
        self.runs_dir = self.state_dir / "runs"
        self.arenas_dir = self.state_dir / "arenas"
        self.migrations_dir = self.state_dir / "migrations"
        self.ensure_dirs()

    def ensure_dirs(self) -> None:
        for path in [
            self.state_dir,
            self.claims_dir,
            self.runs_dir,
            self.arenas_dir,
            self.migrations_dir,
            self.archive_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def _claim_path(self, claim_key: str) -> Path:
        return self.claims_dir / f"{claim_key}.json"

    def _run_path(self, run_id: str) -> Path:
        return self.runs_dir / f"{run_id}.json"

    def _arena_path(self, arena_id: str) -> Path:
        return self.arenas_dir / f"{arena_id}.json"

    def list_claims(self) -> List[Dict[str, Any]]:
        claims: List[Dict[str, Any]] = []
        for path in sorted(self.claims_dir.glob("*.json")):
            try:
                claims.append(_read_json(path))
            except Exception:
                continue
        return claims

    def list_runs(self) -> List[Dict[str, Any]]:
        runs: List[Dict[str, Any]] = []
        for path in sorted(self.runs_dir.glob("*.json")):
            try:
                runs.append(self._decorate_run(_read_json(path)))
            except Exception:
                continue
        runs.sort(
            key=lambda item: (
                item.get("updated_at", ""),
                item.get("created_at", ""),
                item.get("run_id", ""),
            ),
            reverse=True,
        )
        return runs

    def list_arenas(self) -> List[Dict[str, Any]]:
        arenas: List[Dict[str, Any]] = []
        for path in sorted(self.arenas_dir.glob("*.json")):
            try:
                arenas.append(_read_json(path))
            except Exception:
                continue
        arenas.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return arenas

    def get_claim(self, claim_key: str) -> Optional[Dict[str, Any]]:
        path = self._claim_path(claim_key)
        if not path.exists():
            return None
        return _read_json(path)

    def save_claim(self, claim_record: Dict[str, Any]) -> Dict[str, Any]:
        claim_record = dict(claim_record)
        claim_record["batch_tags"] = _dedupe_str_list(claim_record.get("batch_tags", []))
        claim_record["run_ids"] = _dedupe_str_list(claim_record.get("run_ids", []))
        claim_record["updated_at"] = utc_now_iso()
        _atomic_write_json(self._claim_path(claim_record["claim_key"]), claim_record)
        return claim_record

    def get_or_create_claim(
        self,
        text: str,
        *,
        batch_tags: Optional[Iterable[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple[Dict[str, Any], bool]:
        claim_key = claim_key_for_text(text)
        existing = self.get_claim(claim_key)
        normalized_text = normalize_claim_text(text)
        if existing:
            existing["batch_tags"] = _dedupe_str_list([*(existing.get("batch_tags", [])), *(batch_tags or [])])
            existing["metadata"] = {**(existing.get("metadata") or {}), **(metadata or {})}
            return self.save_claim(existing), False

        claim_record = {
            "claim_key": claim_key,
            "text": collapse_whitespace(text),
            "normalized_text": normalized_text,
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "batch_tags": _dedupe_str_list(batch_tags or []),
            "run_ids": [],
            "latest_run_id": None,
            "metadata": metadata or {},
        }
        return self.save_claim(claim_record), True

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        path = self._run_path(run_id)
        if not path.exists():
            return None
        return self._decorate_run(_read_json(path))

    def save_run(self, run_record: Dict[str, Any]) -> Dict[str, Any]:
        run_record = dict(run_record)
        run_record["batch_tags"] = _dedupe_str_list(run_record.get("batch_tags", []))
        run_record["updated_at"] = utc_now_iso()
        run_record.setdefault("usage", empty_usage())
        run_record.setdefault("usage_by_stage", {})
        run_record.setdefault("artifact_paths", {})
        run_record.setdefault("prompt_provenance", default_prompt_provenance())
        _atomic_write_json(self._run_path(run_record["run_id"]), run_record)
        claim = self.get_claim(run_record["claim_key"])
        if claim:
            run_ids = _dedupe_str_list([*(claim.get("run_ids", [])), run_record["run_id"]])
            claim["run_ids"] = run_ids
            claim["batch_tags"] = _dedupe_str_list([*(claim.get("batch_tags", [])), *(run_record.get("batch_tags", []))])
            latest_run = self.get_run(claim.get("latest_run_id", "")) if claim.get("latest_run_id") else None
            if latest_run is None or latest_run.get("updated_at", "") <= run_record.get("updated_at", ""):
                claim["latest_run_id"] = run_record["run_id"]
            self.save_claim(claim)
        return self._decorate_run(run_record)

    def create_run(
        self,
        *,
        claim_record: Dict[str, Any],
        batch_tags: Optional[Iterable[str]] = None,
        arena_id: Optional[str] = None,
        execution_mode: str = "full_pipeline",
        stop_after: str = "final_report",
        provider_snapshot: Optional[Dict[str, Any]] = None,
        model_overrides: Optional[Dict[str, str]] = None,
        search_config: Optional[Dict[str, Any]] = None,
        bibliometric_config: Optional[Dict[str, Any]] = None,
        cost_estimate: Optional[Dict[str, Any]] = None,
        cost_confirmation: Optional[Dict[str, Any]] = None,
        transport_batch_id: Optional[str] = None,
        transport_claim_id: Optional[str] = None,
        review_type: str = "regular",
        status: str = "queued",
        source: str = "submit",
        legacy_lookup: Optional[Dict[str, str]] = None,
        initial_claim_data: Optional[Dict[str, Any]] = None,
        completed_stage: Optional[str] = None,
        is_stage_checkpoint: Optional[bool] = None,
        seed_from_run_id: Optional[str] = None,
        candidate_id: Optional[str] = None,
        candidate_index: Optional[int] = None,
        candidate_prefix: Optional[str] = None,
        candidate_label: Optional[str] = None,
        candidate_color: Optional[str] = None,
        prompt_provenance: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        run_id = transport_claim_id or uuid.uuid4().hex[:12]
        created_at = utc_now_iso()
        provider_snapshot = dict(provider_snapshot or {})
        stop_after = normalize_stop_after(stop_after)
        if provider_snapshot.get("default_model") and not provider_snapshot.get("task_defaults"):
            provider_snapshot["task_defaults"] = {"default": provider_snapshot["default_model"]}
        run_record = {
            "run_id": run_id,
            "claim_key": claim_record["claim_key"],
            "text": claim_record["text"],
            "normalized_text": claim_record["normalized_text"],
            "created_at": created_at,
            "updated_at": created_at,
            "status": status,
            "review_type": review_type,
            "source": source,
            "execution_mode": execution_mode,
            "stop_after": stop_after,
            "completed_stage": completed_stage,
            "is_stage_checkpoint": (
                bool(is_stage_checkpoint)
                if is_stage_checkpoint is not None
                else stage_is_checkpoint(stop_after)
            ),
            "seed_from_run_id": seed_from_run_id,
            "candidate_id": candidate_id,
            "candidate_index": candidate_index,
            "candidate_prefix": candidate_prefix,
            "candidate_label": candidate_label,
            "candidate_color": candidate_color,
            "batch_tags": _dedupe_str_list(batch_tags or []),
            "arena_id": arena_id,
            "provider_snapshot": provider_snapshot,
            "model_overrides": dict(model_overrides or {}),
            "search_config": dict(search_config or {"num_queries": 5, "results_per_query": 5}),
            "bibliometric_config": dict(
                bibliometric_config
                or {
                    "use_bibliometrics": True,
                    "author_impact_weight": 0.4,
                    "citation_impact_weight": 0.4,
                    "venue_impact_weight": 0.2,
                }
            ),
            "cost_estimate": dict(cost_estimate or {}),
            "cost_confirmation": dict(cost_confirmation or {}),
            "usage": empty_usage(),
            "usage_by_stage": {},
            "report": None,
            "report_available": False,
            "prompt_provenance": dict(prompt_provenance or default_prompt_provenance()),
            "transport": {
                "batch_id": transport_batch_id or arena_id or uuid.uuid4().hex[:8],
                "claim_id": transport_claim_id or run_id,
            },
            "legacy_lookup": dict(legacy_lookup or {}),
            "artifact_paths": {},
            "claim_data": dict(initial_claim_data or {}),
        }
        return self.save_run(run_record)

    def save_arena(self, arena_record: Dict[str, Any]) -> Dict[str, Any]:
        arena_record = dict(arena_record)
        arena_record["updated_at"] = utc_now_iso()
        arena_record["claim_keys"] = _dedupe_str_list(arena_record.get("claim_keys", []))
        arena_record["run_ids"] = _dedupe_str_list(arena_record.get("run_ids", []))
        arena_record["current_stage"] = normalize_stop_after(arena_record.get("current_stage"))
        arena_record["stage_history"] = list(arena_record.get("stage_history") or [])
        _atomic_write_json(self._arena_path(arena_record["arena_id"]), arena_record)
        return arena_record

    def create_arena(
        self,
        *,
        title: str,
        batch_tags: Optional[Iterable[str]] = None,
        execution_mode: str = "full_pipeline",
        current_stage: str = "final_report",
        candidates: Optional[List[Dict[str, Any]]] = None,
        claim_keys: Optional[Iterable[str]] = None,
        run_ids: Optional[Iterable[str]] = None,
        stage_history: Optional[List[Dict[str, Any]]] = None,
        source: str = "arena_submit",
    ) -> Dict[str, Any]:
        created_at = utc_now_iso()
        arena_record = {
            "arena_id": uuid.uuid4().hex[:12],
            "title": title,
            "created_at": created_at,
            "updated_at": created_at,
            "batch_tags": _dedupe_str_list(batch_tags or []),
            "execution_mode": execution_mode,
            "current_stage": normalize_stop_after(current_stage),
            "candidates": list(candidates or []),
            "claim_keys": _dedupe_str_list(claim_keys or []),
            "run_ids": _dedupe_str_list(run_ids or []),
            "stage_history": list(stage_history or []),
            "source": source,
        }
        return self.save_arena(arena_record)

    def append_arena_stage_history(
        self,
        arena_record: Dict[str, Any],
        *,
        stage: str,
        run_ids: Iterable[str],
        continue_decisions: Optional[List[Dict[str, Any]]] = None,
        source: str,
    ) -> Dict[str, Any]:
        arena_record = dict(arena_record)
        history = list(arena_record.get("stage_history") or [])
        normalized_run_ids = _dedupe_str_list(run_ids)
        history.append(
            {
                "stage": normalize_stop_after(stage),
                "run_ids": normalized_run_ids,
                "continue_decisions": list(continue_decisions or []),
                "source": source,
                "created_at": utc_now_iso(),
            }
        )
        arena_record["current_stage"] = normalize_stop_after(stage)
        arena_record["stage_history"] = history
        arena_record["run_ids"] = _dedupe_str_list([*(arena_record.get("run_ids", [])), *normalized_run_ids])
        return self.save_arena(arena_record)

    def get_arena(self, arena_id: str) -> Optional[Dict[str, Any]]:
        path = self._arena_path(arena_id)
        if not path.exists():
            return None
        arena = _read_json(path)
        run_map = {run["run_id"]: run for run in self.list_runs() if run.get("arena_id") == arena_id}
        stage_history = list(arena.get("stage_history") or [])
        current_run_ids = list(arena.get("run_ids", []))
        if stage_history:
            current_run_ids = list(stage_history[-1].get("run_ids") or current_run_ids)
        claim_groups: Dict[str, Dict[str, Any]] = {}
        for run_id in current_run_ids:
            run = run_map.get(run_id) or self.get_run(run_id)
            if not run:
                continue
            claim_key = run["claim_key"]
            claim_entry = claim_groups.setdefault(
                claim_key,
                {
                    "claim_key": claim_key,
                    "text": run.get("text", ""),
                    "runs": [],
                },
            )
            claim_entry["runs"].append(self.build_enhanced_run_summary(run, include_claim_data=True))
        arena["claim_groups"] = sorted(
            claim_groups.values(),
            key=lambda item: (item.get("text", ""), item.get("claim_key", "")),
        )
        arena["current_stage"] = normalize_stop_after(arena.get("current_stage"))
        history_with_runs = []
        for entry in stage_history:
            run_ids = list(entry.get("run_ids") or [])
            history_runs = []
            for run_id in run_ids:
                run = run_map.get(run_id) or self.get_run(run_id)
                if run:
                    history_runs.append(self.build_enhanced_run_summary(run))
            history_with_runs.append(
                {
                    **entry,
                    "stage_label": stage_label(entry.get("stage")),
                    "runs": history_runs,
                }
            )
        arena["stage_history"] = history_with_runs
        arena["candidate_count"] = len(arena.get("candidates") or [])
        arena["claim_count"] = len(arena.get("claim_keys") or [])
        arena["summary"] = self.build_arena_summary(arena)
        return arena

    def build_arena_summary(self, arena_record: Dict[str, Any]) -> Dict[str, Any]:
        arena_id = arena_record.get("arena_id")
        stage_history = list(arena_record.get("stage_history") or [])
        current_run_ids = list(arena_record.get("run_ids") or [])
        if stage_history:
            current_run_ids = list(stage_history[-1].get("run_ids") or current_run_ids)
        current_runs = [self.get_run(run_id) for run_id in current_run_ids]
        current_runs = [run for run in current_runs if run]

        status_counts = Counter(str(run.get("status", "unknown")) for run in current_runs)
        if current_runs and all(str(run.get("status")) == "processed" for run in current_runs):
            status = "completed" if arena_record.get("current_stage") == "final_report" else "ready_for_review"
        elif current_runs:
            status = "in_progress"
        else:
            status = "empty"

        expected_cost = 0.0
        actual_cost = 0.0
        updated_at = arena_record.get("updated_at")
        for run in current_runs:
            estimate = (run.get("cost_estimate") or {}).get("expected") or {}
            usage = run.get("usage") or {}
            try:
                expected_cost += float(estimate.get("cost_usd", 0.0) or 0.0)
            except Exception:
                pass
            try:
                actual_cost += float(usage.get("cost_usd", 0.0) or 0.0)
            except Exception:
                pass
            stamp = run.get("updated_at")
            if stamp and (updated_at is None or stamp > updated_at):
                updated_at = stamp

        return {
            "arena_id": arena_id,
            "title": arena_record.get("title") or arena_id,
            "created_at": arena_record.get("created_at"),
            "updated_at": updated_at,
            "claim_count": len(arena_record.get("claim_keys") or []),
            "candidate_count": len(arena_record.get("candidates") or []),
            "current_stage": normalize_stop_after(arena_record.get("current_stage")),
            "current_stage_label": stage_label(arena_record.get("current_stage")),
            "status": status,
            "status_counts": dict(status_counts),
            "expected_cost_usd": round(expected_cost, 6),
            "actual_cost_usd": round(actual_cost, 6),
            "batch_tags": list(arena_record.get("batch_tags") or []),
            "execution_mode": arena_record.get("execution_mode", "full_pipeline"),
        }

    def list_arenas_summary(self, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        summaries = [self.build_arena_summary(arena) for arena in self.list_arenas()]
        summaries.sort(key=lambda item: (item.get("updated_at") or "", item.get("created_at") or ""), reverse=True)
        if isinstance(limit, int) and limit > 0:
            return summaries[:limit]
        return summaries

    def build_arena_progress(self, arena_id: str) -> Optional[Dict[str, Any]]:
        arena = self.get_arena(arena_id)
        if not arena:
            return None

        candidate_rollups: Dict[str, Dict[str, Any]] = {}
        for group in arena.get("claim_groups") or []:
            for run in group.get("runs") or []:
                candidate = run.get("candidate") or self._candidate_identity(run)
                candidate_id = candidate.get("candidate_id") or f"{candidate.get('prefix')}::{run.get('provider_id')}"
                rollup = candidate_rollups.setdefault(
                    candidate_id,
                    {
                        "candidate": candidate,
                        "provider_id": run.get("provider_id"),
                        "provider_label": run.get("provider_label"),
                        "default_model": run.get("default_model"),
                        "effective_models": run.get("effective_models") or {},
                        "status_counts": Counter(),
                        "current_stage_counts": Counter(),
                        "run_count": 0,
                        "completed_runs": 0,
                        "expected_cost_usd": 0.0,
                        "actual_cost_usd": 0.0,
                        "stage_timings_ms": defaultdict(int),
                        "total_elapsed_ms": 0,
                        "issue_count": 0,
                        "retry_count": 0,
                        "truncation_count": 0,
                        "context_overflow_count": 0,
                        "inaccessible_papers_count": 0,
                        "waiting_on_run_id": run.get("waiting_on_run_id"),
                        "current_stage_statuses": Counter(),
                        "runs": [],
                    },
                )
                health = run.get("quality_health") or {}
                rollup["run_count"] += 1
                rollup["status_counts"][run.get("status", "unknown")] += 1
                rollup["current_stage_counts"][run.get("current_stage", "unknown")] += 1
                rollup["current_stage_statuses"][run.get("current_stage_status", "unknown")] += 1
                if run.get("status") == "processed":
                    rollup["completed_runs"] += 1
                estimate = ((run.get("cost_estimate") or {}).get("expected") or {}).get("cost_usd", 0.0)
                usage_cost = (run.get("usage") or {}).get("cost_usd", 0.0)
                rollup["expected_cost_usd"] += float(estimate or 0.0)
                rollup["actual_cost_usd"] += float(usage_cost or 0.0)
                for stage_name, value in (run.get("stage_timings_ms") or {}).items():
                    rollup["stage_timings_ms"][stage_name] += int(value or 0)
                rollup["total_elapsed_ms"] += int(run.get("total_elapsed_ms", 0) or 0)
                rollup["issue_count"] += int(health.get("issues_count", 0) or 0)
                rollup["retry_count"] += int(health.get("retry_count", 0) or 0)
                rollup["truncation_count"] += int(health.get("truncation_count", 0) or 0)
                rollup["context_overflow_count"] += int(health.get("context_overflow_count", 0) or 0)
                rollup["inaccessible_papers_count"] += int(health.get("inaccessible_papers_count", 0) or 0)
                rollup["runs"].append(
                    {
                        "run_id": run.get("run_id"),
                        "claim_key": run.get("claim_key"),
                        "status": run.get("status"),
                        "current_stage": run.get("current_stage"),
                        "current_stage_status": run.get("current_stage_status"),
                    }
                )

        candidates = []
        for candidate_id, rollup in candidate_rollups.items():
            current_stage = rollup["current_stage_counts"].most_common(1)[0][0] if rollup["current_stage_counts"] else "query_generation"
            current_stage_status = (
                rollup["current_stage_statuses"].most_common(1)[0][0] if rollup["current_stage_statuses"] else "queued"
            )
            candidate_status = (
                "completed"
                if rollup["completed_runs"] == rollup["run_count"] and rollup["run_count"] > 0
                else current_stage_status
            )
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "candidate": rollup["candidate"],
                    "provider_id": rollup["provider_id"],
                    "provider_label": rollup["provider_label"],
                    "default_model": rollup["default_model"],
                    "effective_models": rollup["effective_models"],
                    "status": candidate_status,
                    "status_counts": dict(rollup["status_counts"]),
                    "current_stage": current_stage,
                    "current_stage_label": stage_label(current_stage),
                    "current_stage_status": current_stage_status,
                    "run_count": rollup["run_count"],
                    "completed_runs": rollup["completed_runs"],
                    "expected_cost_usd": round(rollup["expected_cost_usd"], 6),
                    "actual_cost_usd": round(rollup["actual_cost_usd"], 6),
                    "stage_timings_ms": dict(rollup["stage_timings_ms"]),
                    "total_elapsed_ms": rollup["total_elapsed_ms"],
                    "issue_count": rollup["issue_count"],
                    "retry_count": rollup["retry_count"],
                    "truncation_count": rollup["truncation_count"],
                    "context_overflow_count": rollup["context_overflow_count"],
                    "inaccessible_papers_count": rollup["inaccessible_papers_count"],
                    "waiting_on_run_id": rollup["waiting_on_run_id"],
                    "dependency_state": (
                        "waiting_for_baseline" if current_stage_status == "waiting_for_baseline" else "independent"
                    ),
                    "runs": rollup["runs"],
                }
            )

        candidates.sort(key=lambda item: item["candidate"]["index"])
        return {
            "arena_id": arena_id,
            "summary": self.build_arena_summary(arena),
            "candidates": candidates,
        }

    @staticmethod
    def effective_models(run_record: Dict[str, Any]) -> Dict[str, Any]:
        provider_snapshot = run_record.get("provider_snapshot") or {}
        task_defaults = provider_snapshot.get("task_defaults") or {}
        model_overrides = run_record.get("model_overrides") or {}
        default_model = provider_snapshot.get("default_model")
        task_models: Dict[str, str] = {}
        for task_name in ["query_generation", "paper_analysis", "venue_scoring", "final_report"]:
            task_models[task_name] = (
                model_overrides.get(task_name)
                or task_defaults.get(task_name)
                or task_defaults.get("default")
                or default_model
                or "unknown-model"
            )
        return {
            "default_model": default_model,
            "task_defaults": dict(task_defaults),
            "overrides": dict(model_overrides),
            "task_models": task_models,
        }

    @staticmethod
    def _candidate_identity(run_record: Dict[str, Any]) -> Dict[str, Any]:
        index = run_record.get("candidate_index")
        if not isinstance(index, int) or index < 0:
            index = 0
        prefix = run_record.get("candidate_prefix") or candidate_prefix_for_index(index)
        provider_snapshot = run_record.get("provider_snapshot") or {}
        label = (
            run_record.get("candidate_label")
            or provider_snapshot.get("label")
            or provider_snapshot.get("default_model")
            or f"Candidate {prefix}"
        )
        return {
            "candidate_id": run_record.get("candidate_id"),
            "index": index,
            "prefix": prefix,
            "label": label,
            "color": run_record.get("candidate_color") or candidate_color_for_index(index),
        }

    def _artifact_file_candidates(self, run_record: Dict[str, Any], artifact_dir: str, extension: str) -> List[Path]:
        artifact_paths = run_record.get("artifact_paths") or {}
        direct_key = {
            "traces": "trace_file",
            "issues": "issues_file",
        }.get(artifact_dir)
        direct = artifact_paths.get(direct_key) if direct_key else None
        candidates: List[Path] = []
        if direct:
            candidates.append(Path(str(direct)))

        transport = run_record.get("transport") or {}
        batch_id = transport.get("batch_id")
        claim_id = transport.get("claim_id")
        if not batch_id or not claim_id:
            return candidates

        for root in [self.trace_dir, self.archive_dir]:
            if root == self.trace_dir:
                base = root / batch_id / artifact_dir
            else:
                base = root / "traces" / batch_id / artifact_dir
            candidates.append(base / f"{claim_id}.{extension}")
            candidates.append(base / f"{claim_id}.{extension}.gz")
        return candidates

    def _claim_file_candidates(self, run_record: Dict[str, Any]) -> List[Path]:
        artifact_paths = run_record.get("artifact_paths") or {}
        direct_candidates = []
        for key in ["saved_jobs_file", "queued_jobs_file"]:
            value = artifact_paths.get(key)
            if value:
                direct_candidates.append(Path(str(value)))

        transport = run_record.get("transport") or {}
        batch_id = transport.get("batch_id")
        claim_id = transport.get("claim_id")
        if not batch_id or not claim_id:
            return direct_candidates

        return [
            *direct_candidates,
            self.saved_jobs_dir / batch_id / f"{claim_id}.txt",
            self.queued_jobs_dir / batch_id / f"{claim_id}.txt",
            self.archive_dir / "saved_jobs" / batch_id / f"{claim_id}.txt",
            self.archive_dir / "queued_jobs" / batch_id / f"{claim_id}.txt",
        ]

    def _locate_existing_file(self, candidates: Iterable[Path]) -> Optional[Path]:
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _load_issue_records(self, run_record: Dict[str, Any]) -> List[Dict[str, Any]]:
        path = self._locate_existing_file(self._artifact_file_candidates(run_record, "issues", "jsonl"))
        return _read_jsonl(path) if path else []

    def _load_trace_records(self, run_record: Dict[str, Any]) -> List[Dict[str, Any]]:
        path = self._locate_existing_file(self._artifact_file_candidates(run_record, "traces", "jsonl"))
        return _read_jsonl(path) if path else []

    @staticmethod
    def _trace_summary(trace_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        retries = 0
        truncation_count = 0
        total_tokens = 0
        models = set()
        for record in trace_records:
            status = str(record.get("status", "")).lower()
            if status == "retrying":
                retries += 1
            raw_output = str(record.get("raw_output") or "")
            if "...[truncated]" in raw_output:
                truncation_count += 1
            usage = record.get("usage") or {}
            try:
                total_tokens += int(usage.get("total_tokens", 0) or 0)
            except Exception:
                pass
            model_name = record.get("model_used") or record.get("model_requested")
            if model_name:
                models.add(str(model_name))
        return {
            "retry_count": retries,
            "truncation_count": truncation_count,
            "models_used": sorted(models),
            "total_tokens": total_tokens,
        }

    @staticmethod
    def _stage_timings_from_traces(trace_records: List[Dict[str, Any]]) -> Dict[str, int]:
        timings: Dict[str, int] = {stage: 0 for stage in ["query_generation", "paper_analysis", "venue_scoring", "final_report"]}
        for record in trace_records:
            stage = str(record.get("stage") or "").strip()
            if stage not in timings:
                continue
            try:
                timings[stage] += int(record.get("latency_ms", 0) or 0)
            except Exception:
                continue
        return timings

    def _merge_prompt_provenance(
        self,
        run_record: Dict[str, Any],
        claim_data: Dict[str, Any],
        trace_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        provenance = default_prompt_provenance()

        for source in [run_record.get("prompt_provenance") or {}, claim_data.get("prompt_provenance") or {}]:
            for stage, entry in source.items():
                if not isinstance(entry, dict):
                    continue
                current = provenance.setdefault(stage, stage_prompt_provenance(stage))
                current.update({key: value for key, value in entry.items() if value is not None})

        for record in trace_records:
            stage = str(record.get("stage") or "").strip()
            if stage not in provenance:
                continue
            entry = provenance[stage]
            if entry.get("rendered_prompt_hash"):
                continue
            messages = record.get("messages") or []
            if not isinstance(messages, list):
                continue
            system_prompt = "\n".join(
                str(message.get("content") or "")
                for message in messages
                if isinstance(message, dict) and message.get("role") == "system"
            )
            user_prompt = "\n".join(
                str(message.get("content") or "")
                for message in messages
                if isinstance(message, dict) and message.get("role") == "user"
            )
            if system_prompt or user_prompt:
                entry["rendered_prompt_hash"] = sha256_text(f"{system_prompt}\n{user_prompt}")
        return provenance

    def _infer_current_stage(
        self,
        run_record: Dict[str, Any],
        claim_data: Dict[str, Any],
    ) -> Tuple[str, str]:
        status = str(run_record.get("status", "unknown") or "unknown").strip().lower()
        completed_stage = run_record.get("completed_stage")
        if status == "processed":
            final_stage = completed_stage or normalize_stop_after(run_record.get("stop_after"))
            return final_stage, "completed"
        if status == "waiting_for_baseline":
            return "query_generation", "waiting_for_baseline"
        if status == "ready_for_search":
            return "paper_analysis", "retrieving_papers"
        if status == "ready_for_analysis":
            processed_papers = claim_data.get("processed_papers") or []
            if processed_papers:
                if any(paper.get("score", -1) == -1 for paper in processed_papers if isinstance(paper, dict)):
                    return "venue_scoring", "scoring_evidence"
                return "final_report", "drafting_report"
            if (claim_data.get("raw_papers") or []) or (claim_data.get("inaccessible_papers") or []):
                return "paper_analysis", "analyzing_evidence"
            return "paper_analysis", "queued"
        return RUN_STATUS_TO_STAGE.get(status, "query_generation"), status

    def _quality_health(
        self,
        run_record: Dict[str, Any],
        claim_data: Dict[str, Any],
        issue_records: List[Dict[str, Any]],
        trace_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        trace_summary = self._trace_summary(trace_records)
        context_overflow_count = sum(
            1
            for issue in issue_records
            if "context overflow prevented" in str(issue.get("message", "")).lower()
        )
        truncation_count = trace_summary["truncation_count"] + sum(
            1
            for issue in issue_records
            if "truncat" in str(issue.get("message", "")).lower()
        )
        inaccessible_count = len(claim_data.get("inaccessible_papers") or [])
        return {
            "issues_count": len(issue_records),
            "retry_count": trace_summary["retry_count"],
            "truncation_count": truncation_count,
            "truncation_flag": truncation_count > 0,
            "context_overflow_count": context_overflow_count,
            "context_overflow_flag": context_overflow_count > 0,
            "inaccessible_papers_count": inaccessible_count,
            "warning_count": sum(
                1 for issue in issue_records if str(issue.get("severity", "")).upper() == "WARN"
            ),
            "error_count": sum(
                1 for issue in issue_records if str(issue.get("severity", "")).upper() == "ERROR"
            ),
        }

    @staticmethod
    def _last_activity_at(
        run_record: Dict[str, Any],
        claim_file: Optional[Path],
        trace_records: List[Dict[str, Any]],
    ) -> Optional[str]:
        timestamps = [run_record.get("updated_at")]
        if claim_file:
            try:
                timestamps.append(datetime.fromtimestamp(claim_file.stat().st_mtime, tz=timezone.utc).isoformat())
            except OSError:
                pass
        for record in trace_records:
            for key in ["timestamp_end", "timestamp_start"]:
                value = record.get(key)
                if value:
                    timestamps.append(str(value))
        filtered = [value for value in timestamps if value]
        return max(filtered) if filtered else None

    def build_enhanced_run_summary(
        self,
        run_record: Dict[str, Any],
        *,
        include_claim_data: bool = False,
    ) -> Dict[str, Any]:
        summary = self.build_run_summary(run_record)
        claim_data = self.load_claim_data_for_run(run_record) or run_record.get("claim_data") or {}
        if not isinstance(claim_data, dict):
            claim_data = {}
        trace_records = self._load_trace_records(run_record)
        issue_records = self._load_issue_records(run_record)
        claim_file = self._locate_existing_file(self._claim_file_candidates(run_record))
        current_stage, current_stage_status = self._infer_current_stage(run_record, claim_data)
        stage_timings_ms = self._stage_timings_from_traces(trace_records)
        total_elapsed_ms = sum(stage_timings_ms.values())

        summary.update(
            {
                "candidate": self._candidate_identity(run_record),
                "current_stage": current_stage,
                "current_stage_label": stage_label(current_stage),
                "current_stage_status": current_stage_status,
                "waiting_on_run_id": run_record.get("reuse_from_run_id"),
                "stage_timings_ms": stage_timings_ms,
                "total_elapsed_ms": total_elapsed_ms,
                "last_activity_at": self._last_activity_at(run_record, claim_file, trace_records),
                "prompt_provenance": self._merge_prompt_provenance(run_record, claim_data, trace_records),
                "quality_health": self._quality_health(run_record, claim_data, issue_records, trace_records),
                "trace_summary": self._trace_summary(trace_records),
            }
        )
        if include_claim_data:
            summary["claim_data"] = claim_data
            summary["issue_records"] = issue_records
        return summary

    def build_run_summary(self, run_record: Dict[str, Any]) -> Dict[str, Any]:
        report = run_record.get("report") or {}
        usage = report.get("usage_summary") or run_record.get("usage") or empty_usage()
        transport = run_record.get("transport") or {}
        provider_snapshot = run_record.get("provider_snapshot") or {}
        stop_after = normalize_stop_after(run_record.get("stop_after"))
        completed_stage = run_record.get("completed_stage")
        candidate = self._candidate_identity(run_record)
        return {
            "run_id": run_record.get("run_id"),
            "claim_key": run_record.get("claim_key"),
            "text": run_record.get("text", ""),
            "status": run_record.get("status", "unknown"),
            "review_type": run_record.get("review_type", "regular"),
            "batch_tags": run_record.get("batch_tags", []),
            "arena_id": run_record.get("arena_id"),
            "execution_mode": run_record.get("execution_mode", "full_pipeline"),
            "stop_after": stop_after,
            "completed_stage": completed_stage,
            "completed_stage_label": stage_label(completed_stage),
            "is_stage_checkpoint": bool(run_record.get("is_stage_checkpoint")),
            "seed_from_run_id": run_record.get("seed_from_run_id"),
            "continue_to_stage": continue_stage_for_run(run_record),
            "source": run_record.get("source"),
            "provider_id": provider_snapshot.get("provider_id"),
            "provider_label": provider_snapshot.get("label"),
            "provider_type": provider_snapshot.get("provider_type"),
            "default_model": provider_snapshot.get("default_model"),
            "provider_snapshot": provider_snapshot,
            "candidate_id": candidate["candidate_id"],
            "candidate_index": candidate["index"],
            "candidate_prefix": candidate["prefix"],
            "candidate_label": candidate["label"],
            "candidate_color": candidate["color"],
            "effective_models": self.effective_models(run_record),
            "model_overrides": run_record.get("model_overrides") or {},
            "search_config": run_record.get("search_config") or {},
            "bibliometric_config": run_record.get("bibliometric_config") or {},
            "cost_estimate": run_record.get("cost_estimate") or {},
            "cost_confirmation": run_record.get("cost_confirmation") or {},
            "prompt_provenance": run_record.get("prompt_provenance") or default_prompt_provenance(),
            "claimRating": report.get("claimRating"),
            "rating_label": self.rating_label(report.get("claimRating")),
            "report_available": bool(run_record.get("report_available")),
            "checkpoint_complete": checkpoint_complete(run_record),
            "usage": usage,
            "usage_by_stage": run_record.get("usage_by_stage") or {},
            "issues_count": len((report.get("issues") or [])),
            "reuse_from_run_id": run_record.get("reuse_from_run_id"),
            "transport_batch_id": transport.get("batch_id"),
            "transport_claim_id": transport.get("claim_id"),
            "created_at": run_record.get("created_at"),
            "updated_at": run_record.get("updated_at"),
            "location": run_record.get("location"),
        }

    def hydrate_run(self, run_record: Dict[str, Any]) -> Dict[str, Any]:
        summary = self.build_run_summary(run_record)
        claim_data = run_record.get("claim_data") or self.load_claim_data_for_run(run_record) or {}
        report = claim_data.get("report") if isinstance(claim_data, dict) else None
        report_available = isinstance(report, dict) and bool(report)
        summary["claim_id"] = summary.get("transport_claim_id") or summary["run_id"]
        summary["report_available"] = bool(summary.get("report_available") or report_available)
        summary["claim_location"] = run_record.get("location") or (
            "saved_jobs" if summary.get("status") == "processed" else "queued_jobs"
        )
        summary["location"] = summary["claim_location"]
        summary["report"] = report if isinstance(report, dict) else (run_record.get("report") or {})
        summary["claim_data"] = (
            claim_data
            if isinstance(claim_data, dict) and claim_data
            else {
                "text": summary.get("text", ""),
                "status": summary.get("status", "unknown"),
                "report": summary.get("report", {}),
                "claim_id": summary["claim_id"],
                "batch_id": summary.get("transport_batch_id"),
            }
        )
        summary["review_value"] = review_value_for_stage(summary["claim_data"], summary.get("completed_stage"))
        return summary

    @staticmethod
    def rating_label(rating: Optional[int]) -> str:
        labels = {
            0: "No Evidence",
            1: "Contradicted",
            2: "Likely False",
            3: "Mixed Evidence",
            4: "Likely True",
            5: "Highly Supported",
        }
        return labels.get(rating, "Unrated")

    def list_runs_for_claim(self, claim_key: str) -> List[Dict[str, Any]]:
        return [run for run in self.list_runs() if run.get("claim_key") == claim_key]

    def find_claim_by_text(self, text: str) -> Optional[Dict[str, Any]]:
        return self.get_claim(claim_key_for_text(text))

    def find_run_by_legacy(self, batch_id: str, claim_id: str) -> Optional[Dict[str, Any]]:
        for run in self.list_runs():
            transport = run.get("transport") or {}
            legacy = run.get("legacy_lookup") or {}
            if transport.get("batch_id") == batch_id and transport.get("claim_id") == claim_id:
                return run
            if legacy.get("batch_id") == batch_id and legacy.get("claim_id") == claim_id:
                return run
        return None

    def resolve_report_path(self, run_record: Dict[str, Any]) -> Optional[Path]:
        return self._locate_existing_file(self._claim_file_candidates(run_record))

    def load_claim_data_for_run(self, run_record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        report_path = self.resolve_report_path(run_record)
        if not report_path:
            return None
        try:
            return _read_json(report_path)
        except Exception:
            return None

    def materialize_run_to_queue(
        self,
        run_record: Dict[str, Any],
        *,
        status: Optional[str] = None,
        seeded_claim_data: Optional[Dict[str, Any]] = None,
    ) -> Path:
        transport = run_record.get("transport") or {}
        batch_id = transport["batch_id"]
        claim_id = transport["claim_id"]
        claim_file = self.queued_jobs_dir / batch_id / f"{claim_id}.txt"
        claim_file.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "text": run_record.get("text", ""),
            "status": status or run_record.get("status", "queued"),
            "batch_id": batch_id,
            "claim_id": claim_id,
            "run_id": run_record.get("run_id"),
            "claim_key": run_record.get("claim_key"),
            "batch_tags": run_record.get("batch_tags", []),
            "arena_id": run_record.get("arena_id"),
            "review_type": run_record.get("review_type", "regular"),
            "execution_mode": run_record.get("execution_mode", "full_pipeline"),
            "stop_after": run_record.get("stop_after", "final_report"),
            "completed_stage": run_record.get("completed_stage"),
            "is_stage_checkpoint": bool(run_record.get("is_stage_checkpoint")),
            "seed_from_run_id": run_record.get("seed_from_run_id"),
            "candidate_id": run_record.get("candidate_id"),
            "candidate_index": run_record.get("candidate_index"),
            "candidate_prefix": run_record.get("candidate_prefix"),
            "candidate_label": run_record.get("candidate_label"),
            "candidate_color": run_record.get("candidate_color"),
            "provider_snapshot": run_record.get("provider_snapshot") or {},
            "model_overrides": run_record.get("model_overrides") or {},
            "search_config": run_record.get("search_config") or {},
            "bibliometric_config": run_record.get("bibliometric_config") or {},
            "cost_estimate": run_record.get("cost_estimate") or {},
            "cost_confirmation": run_record.get("cost_confirmation") or {},
            "prompt_provenance": run_record.get("prompt_provenance") or default_prompt_provenance(),
            "usage": run_record.get("usage") or empty_usage(),
            "usage_by_stage": run_record.get("usage_by_stage") or {},
            "additional_info": "",
        }
        if seeded_claim_data:
            payload.update(seeded_claim_data)
            payload["batch_id"] = batch_id
            payload["claim_id"] = claim_id
            payload["run_id"] = run_record.get("run_id")
            payload["claim_key"] = run_record.get("claim_key")
            payload["batch_tags"] = run_record.get("batch_tags", [])
            payload["arena_id"] = run_record.get("arena_id")
            payload["provider_snapshot"] = run_record.get("provider_snapshot") or {}
            payload["model_overrides"] = run_record.get("model_overrides") or {}
            payload["search_config"] = run_record.get("search_config") or {}
            payload["bibliometric_config"] = run_record.get("bibliometric_config") or {}
            payload["usage"] = run_record.get("usage") or empty_usage()
            payload["usage_by_stage"] = run_record.get("usage_by_stage") or {}
            payload["stop_after"] = run_record.get("stop_after", "final_report")
            payload["completed_stage"] = run_record.get("completed_stage")
            payload["is_stage_checkpoint"] = bool(run_record.get("is_stage_checkpoint"))
            payload["seed_from_run_id"] = run_record.get("seed_from_run_id")
            payload["candidate_id"] = run_record.get("candidate_id")
            payload["candidate_index"] = run_record.get("candidate_index")
            payload["candidate_prefix"] = run_record.get("candidate_prefix")
            payload["candidate_label"] = run_record.get("candidate_label")
            payload["candidate_color"] = run_record.get("candidate_color")
            payload["prompt_provenance"] = run_record.get("prompt_provenance") or default_prompt_provenance()

        _atomic_write_json(claim_file, payload)
        run_record = dict(run_record)
        run_record["status"] = payload.get("status", "queued")
        run_record.setdefault("artifact_paths", {})
        run_record["artifact_paths"]["queued_jobs_file"] = str(claim_file.resolve())
        self.save_run(run_record)
        return claim_file

    def ingest_transport_artifact(self, batch_id: str, claim_id: str) -> Optional[Dict[str, Any]]:
        run = self.find_run_by_legacy(batch_id, claim_id)
        if not run:
            return None
        claim_file = None
        location = None
        for root_name, root in [("saved_jobs", self.saved_jobs_dir), ("queued_jobs", self.queued_jobs_dir)]:
            candidate = root / batch_id / f"{claim_id}.txt"
            if candidate.exists():
                claim_file = candidate
                location = root_name
                break
        if claim_file is None:
            return None

        claim_data = _read_json(claim_file)
        run["text"] = claim_data.get("text", run.get("text", ""))
        run["status"] = claim_data.get("status", run.get("status", "unknown"))
        run["report"] = claim_data.get("report")
        run["report_available"] = isinstance(claim_data.get("report"), dict) and bool(claim_data.get("report"))
        run["usage"] = claim_data.get("usage") or empty_usage()
        run["usage_by_stage"] = claim_data.get("usage_by_stage") or {}
        run["search_config"] = claim_data.get("search_config") or run.get("search_config") or {}
        run["bibliometric_config"] = claim_data.get("bibliometric_config") or run.get("bibliometric_config") or {}
        run["model_overrides"] = claim_data.get("model_overrides") or run.get("model_overrides") or {}
        run["provider_snapshot"] = claim_data.get("provider_snapshot") or run.get("provider_snapshot") or {}
        run["stop_after"] = claim_data.get("stop_after") or run.get("stop_after") or "final_report"
        run["completed_stage"] = claim_data.get("completed_stage") or run.get("completed_stage")
        run["is_stage_checkpoint"] = bool(
            claim_data.get("is_stage_checkpoint", run.get("is_stage_checkpoint", False))
        )
        run["seed_from_run_id"] = claim_data.get("seed_from_run_id") or run.get("seed_from_run_id")
        run["candidate_id"] = claim_data.get("candidate_id") or run.get("candidate_id")
        run["candidate_index"] = claim_data.get("candidate_index", run.get("candidate_index"))
        run["candidate_prefix"] = claim_data.get("candidate_prefix") or run.get("candidate_prefix")
        run["candidate_label"] = claim_data.get("candidate_label") or run.get("candidate_label")
        run["candidate_color"] = claim_data.get("candidate_color") or run.get("candidate_color")
        run["prompt_provenance"] = claim_data.get("prompt_provenance") or run.get("prompt_provenance") or default_prompt_provenance()
        run["claim_data"] = claim_data
        run["location"] = location
        run.setdefault("artifact_paths", {})
        run["artifact_paths"][f"{location}_file"] = str(claim_file.resolve())
        trace_dir = self.trace_dir / batch_id / "traces"
        issue_dir = self.trace_dir / batch_id / "issues"
        for candidate in [trace_dir / f"{claim_id}.jsonl", trace_dir / f"{claim_id}.jsonl.gz"]:
            if candidate.exists():
                run["artifact_paths"]["trace_file"] = str(candidate.resolve())
                break
        for candidate in [
            self.archive_dir / "traces" / batch_id / "traces" / f"{claim_id}.jsonl",
            self.archive_dir / "traces" / batch_id / "traces" / f"{claim_id}.jsonl.gz",
        ]:
            if candidate.exists():
                run["artifact_paths"]["trace_file"] = str(candidate.resolve())
                break
        for candidate in [issue_dir / f"{claim_id}.jsonl", issue_dir / f"{claim_id}.jsonl.gz"]:
            if candidate.exists():
                run["artifact_paths"]["issues_file"] = str(candidate.resolve())
                break
        for candidate in [
            self.archive_dir / "traces" / batch_id / "issues" / f"{claim_id}.jsonl",
            self.archive_dir / "traces" / batch_id / "issues" / f"{claim_id}.jsonl.gz",
        ]:
            if candidate.exists():
                run["artifact_paths"]["issues_file"] = str(candidate.resolve())
                break
        return self.save_run(run)

    def list_batch_tags(self) -> List[str]:
        seen = set()
        tags: List[str] = []
        for run in self.list_runs():
            for tag in run.get("batch_tags", []):
                if tag not in seen:
                    seen.add(tag)
                    tags.append(tag)
        return sorted(tags)

    def build_batch_state(self, batch_tag: str, *, include_all_runs: bool = False) -> Optional[Dict[str, Any]]:
        tagged_runs = [run for run in self.list_runs() if batch_tag in run.get("batch_tags", [])]
        if not tagged_runs:
            return None

        tagged_runs.sort(key=lambda item: (item.get("updated_at", ""), item.get("run_id", "")), reverse=True)
        if include_all_runs:
            selected_runs = tagged_runs
        else:
            latest_by_claim: Dict[str, Dict[str, Any]] = {}
            for run in tagged_runs:
                latest_by_claim.setdefault(run["claim_key"], run)
            selected_runs = list(latest_by_claim.values())
            selected_runs.sort(key=lambda item: item.get("text", ""))

        claims = []
        counts_by_status: Dict[str, int] = {}
        counts_by_location: Dict[str, int] = {}
        processed_claims = 0
        current_claim_id = None
        oldest = None
        newest = None

        for run in selected_runs:
            summary = self.hydrate_run(run)
            claims.append(summary)
            status = summary.get("status", "unknown") or "unknown"
            counts_by_status[status] = counts_by_status.get(status, 0) + 1
            location = summary.get("claim_location", "unknown")
            counts_by_location[location] = counts_by_location.get(location, 0) + 1
            if summary.get("checkpoint_complete"):
                processed_claims += 1
            if status != "processed" and current_claim_id is None:
                current_claim_id = summary["claim_id"]
            stamp = run.get("updated_at")
            if stamp and (oldest is None or stamp < oldest):
                oldest = stamp
            if stamp and (newest is None or stamp > newest):
                newest = stamp

        status = "completed" if all(run.get("status") == "processed" for run in selected_runs) else "processing"
        return {
            "batch_id": batch_tag,
            "status": status,
            "total_claims": len(claims),
            "processed_claims": processed_claims,
            "counts_by_status": counts_by_status,
            "counts_by_location": counts_by_location,
            "has_active_claims": status != "completed",
            "has_partial_resume": any(run.get("source") == "resume" for run in selected_runs),
            "current_claim_id": current_claim_id,
            "timestamp": oldest,
            "updated_at": newest,
            "claims": claims,
            "errors": [],
            "all_runs": [self.hydrate_run(run) for run in tagged_runs],
        }

    def build_claim_detail(self, claim_key: str, *, run_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        claim = self.get_claim(claim_key)
        if not claim:
            return None
        runs = [self.build_enhanced_run_summary(run, include_claim_data=True) for run in self.list_runs_for_claim(claim_key)]
        runs.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        focused_run = None
        if run_id:
            focused_run = next((run for run in runs if run.get("run_id") == run_id), None)
        if focused_run is None and claim.get("latest_run_id"):
            focused_run = next((run for run in runs if run.get("run_id") == claim.get("latest_run_id")), None)
        if focused_run is None and runs:
            focused_run = runs[0]
        alternative_runs = [run for run in runs if focused_run is None or run.get("run_id") != focused_run.get("run_id")]

        source_context = None
        if focused_run and focused_run.get("arena_id"):
            arena = self.get_arena(focused_run["arena_id"])
            source_context = {
                "type": "arena",
                "arena_id": focused_run.get("arena_id"),
                "arena_title": (arena or {}).get("title"),
                "current_stage": (arena or {}).get("current_stage"),
                "candidate_prefix": focused_run.get("candidate_prefix"),
                "candidate_label": focused_run.get("candidate_label"),
            }
        return {
            "claim": claim,
            "runs": runs,
            "focused_run": focused_run,
            "alternative_runs": alternative_runs,
            "source_context": source_context,
        }

    def delete_run(self, run_id: str) -> bool:
        run = self.get_run(run_id)
        if not run:
            return False
        path = self._run_path(run_id)
        if path.exists():
            path.unlink()
        claim = self.get_claim(run["claim_key"])
        if claim:
            claim["run_ids"] = [value for value in claim.get("run_ids", []) if value != run_id]
            if claim.get("latest_run_id") == run_id:
                claim["latest_run_id"] = claim["run_ids"][0] if claim["run_ids"] else None
            if claim["run_ids"]:
                self.save_claim(claim)
            else:
                claim_path = self._claim_path(run["claim_key"])
                if claim_path.exists():
                    claim_path.unlink()
        return True

    def delete_runs_by_batch_tag(self, batch_tag: str) -> List[str]:
        deleted: List[str] = []
        for run in self.list_runs():
            if batch_tag in run.get("batch_tags", []):
                if self.delete_run(run["run_id"]):
                    deleted.append(run["run_id"])
        return deleted

    def _iter_legacy_claim_files(
        self,
        *,
        batch_id: Optional[str] = None,
    ) -> Iterable[Tuple[str, Path, Path]]:
        for root_name, root in [("saved_jobs", self.saved_jobs_dir), ("queued_jobs", self.queued_jobs_dir)]:
            if not root.exists():
                continue
            batch_dirs = [path for path in root.iterdir() if path.is_dir()]
            if batch_id:
                batch_dirs = [path for path in batch_dirs if path.name == batch_id]
            for batch_dir in sorted(batch_dirs):
                for claim_file in sorted(batch_dir.glob("*.txt")):
                    if claim_file.name == "claims.txt":
                        continue
                    yield root_name, batch_dir, claim_file

    def list_legacy_batches(self) -> List[Dict[str, Any]]:
        imported_counts = Counter()
        for run in self.list_runs():
            legacy = run.get("legacy_lookup") or {}
            batch_id = legacy.get("batch_id")
            if batch_id:
                imported_counts[str(batch_id)] += 1

        grouped: Dict[str, Dict[str, Any]] = {}
        for root_name, batch_dir, claim_file in self._iter_legacy_claim_files():
            entry = grouped.setdefault(
                batch_dir.name,
                {
                    "batch_id": batch_dir.name,
                    "claim_count": 0,
                    "roots": [],
                    "status": "pending",
                    "last_modified_at": None,
                },
            )
            entry["claim_count"] += 1
            if root_name not in entry["roots"]:
                entry["roots"].append(root_name)
            modified_at = datetime.fromtimestamp(claim_file.stat().st_mtime, tz=timezone.utc).isoformat()
            if entry["last_modified_at"] is None or modified_at > entry["last_modified_at"]:
                entry["last_modified_at"] = modified_at

        summaries = []
        for batch_id, entry in grouped.items():
            imported_count = imported_counts.get(batch_id, 0)
            status = "imported" if imported_count else "pending"
            if imported_count and imported_count < entry["claim_count"]:
                status = "partially_imported"
            summaries.append(
                {
                    **entry,
                    "imported_count": imported_count,
                    "status": status,
                }
            )
        summaries.sort(key=lambda item: (item.get("last_modified_at") or "", item["batch_id"]), reverse=True)
        return summaries

    def inspect_legacy_batch(self, batch_id: str) -> Optional[Dict[str, Any]]:
        claims = []
        summary = None
        for candidate in self.list_legacy_batches():
            if candidate["batch_id"] == batch_id:
                summary = candidate
                break
        if summary is None:
            return None

        for root_name, batch_dir, claim_file in self._iter_legacy_claim_files(batch_id=batch_id):
            claim_data = _read_json(claim_file)
            claims.append(
                {
                    "claim_id": claim_file.stem,
                    "source_root": root_name,
                    "status": claim_data.get("status", "unknown"),
                    "text": claim_data.get("text", ""),
                    "completed_stage": claim_data.get("completed_stage"),
                    "review_type": claim_data.get("review_type", "regular"),
                    "has_report": bool(claim_data.get("report")),
                    "updated_at": datetime.fromtimestamp(claim_file.stat().st_mtime, tz=timezone.utc).isoformat(),
                }
            )

        return {
            **summary,
            "claims": claims,
        }

    def discover_legacy_batches(self) -> List[str]:
        return [batch["batch_id"] for batch in self.list_legacy_batches() if batch["status"] == "pending"]

    def migration_status(self) -> Dict[str, Any]:
        summaries = self.list_legacy_batches()
        pending = [batch["batch_id"] for batch in summaries if batch["status"] == "pending"]
        return {
            "state_dir": str(self.state_dir.resolve()),
            "pending_batches": pending,
            "pending_count": len(pending),
            "batches": summaries,
        }

    def _persist_imported_legacy_run(
        self,
        *,
        batch_dir: Path,
        claim_file: Path,
        root_name: str,
        claim_data: Dict[str, Any],
        claim_record: Dict[str, Any],
    ) -> Dict[str, Any]:
        run_record = self.create_run(
            claim_record=claim_record,
            batch_tags=[batch_dir.name],
            execution_mode=claim_data.get("execution_mode", "full_pipeline"),
            stop_after=claim_data.get("stop_after", "final_report"),
            provider_snapshot=claim_data.get("provider_snapshot") or {},
            model_overrides=claim_data.get("model_overrides") or {},
            search_config=claim_data.get("search_config") or {},
            bibliometric_config=claim_data.get("bibliometric_config") or {},
            transport_batch_id=batch_dir.name,
            transport_claim_id=claim_file.stem,
            review_type=claim_data.get("review_type", "regular"),
            status=claim_data.get("status", "queued"),
            source="migration",
            legacy_lookup={"batch_id": batch_dir.name, "claim_id": claim_file.stem},
            initial_claim_data=claim_data,
            completed_stage=claim_data.get("completed_stage"),
            is_stage_checkpoint=bool(claim_data.get("is_stage_checkpoint", False)),
            seed_from_run_id=claim_data.get("seed_from_run_id"),
            candidate_id=claim_data.get("candidate_id"),
            candidate_index=claim_data.get("candidate_index"),
            candidate_prefix=claim_data.get("candidate_prefix"),
            candidate_label=claim_data.get("candidate_label"),
            candidate_color=claim_data.get("candidate_color"),
            prompt_provenance=claim_data.get("prompt_provenance") or default_prompt_provenance(),
        )
        run_record["claim_data"] = claim_data
        run_record["report"] = claim_data.get("report")
        run_record["report_available"] = isinstance(claim_data.get("report"), dict) and bool(claim_data.get("report"))
        run_record["usage"] = claim_data.get("usage") or empty_usage()
        run_record["usage_by_stage"] = claim_data.get("usage_by_stage") or {}
        run_record["location"] = root_name
        run_record.setdefault("artifact_paths", {})
        run_record["artifact_paths"][f"{root_name}_file"] = str(claim_file.resolve())
        trace_dir = self.trace_dir / batch_dir.name / "traces"
        issue_dir = self.trace_dir / batch_dir.name / "issues"
        for candidate in [trace_dir / f"{claim_file.stem}.jsonl", trace_dir / f"{claim_file.stem}.jsonl.gz"]:
            if candidate.exists():
                run_record["artifact_paths"]["trace_file"] = str(candidate.resolve())
                break
        for candidate in [issue_dir / f"{claim_file.stem}.jsonl", issue_dir / f"{claim_file.stem}.jsonl.gz"]:
            if candidate.exists():
                run_record["artifact_paths"]["issues_file"] = str(candidate.resolve())
                break
        return self.save_run(run_record)

    def archive_legacy_batch(self, batch_id: str) -> Dict[str, Any]:
        moved_paths: List[str] = []
        for root_name, root in [("saved_jobs", self.saved_jobs_dir), ("queued_jobs", self.queued_jobs_dir)]:
            source = root / batch_id
            if not source.exists():
                continue
            destination = self.archive_dir / root_name / batch_id
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                shutil.rmtree(destination)
            shutil.move(str(source), str(destination))
            moved_paths.append(str(destination))

        trace_source = self.trace_dir / batch_id
        if trace_source.exists():
            trace_destination = self.archive_dir / "traces" / batch_id
            trace_destination.parent.mkdir(parents=True, exist_ok=True)
            if trace_destination.exists():
                shutil.rmtree(trace_destination)
            shutil.move(str(trace_source), str(trace_destination))
            moved_paths.append(str(trace_destination))

        for run in self.list_runs():
            legacy = run.get("legacy_lookup") or {}
            if legacy.get("batch_id") != batch_id:
                continue
            claim_id = legacy.get("claim_id") or (run.get("transport") or {}).get("claim_id")
            if not claim_id:
                continue
            for root_name in ["saved_jobs", "queued_jobs"]:
                archived_claim = self.archive_dir / root_name / batch_id / f"{claim_id}.txt"
                if archived_claim.exists():
                    run.setdefault("artifact_paths", {})
                    run["artifact_paths"][f"{root_name}_file"] = str(archived_claim.resolve())
                    run["location"] = f"archived_{root_name}"
            for artifact_dir, key in [("traces", "trace_file"), ("issues", "issues_file")]:
                for candidate in [
                    self.archive_dir / "traces" / batch_id / artifact_dir / f"{claim_id}.jsonl",
                    self.archive_dir / "traces" / batch_id / artifact_dir / f"{claim_id}.jsonl.gz",
                ]:
                    if candidate.exists():
                        run.setdefault("artifact_paths", {})
                        run["artifact_paths"][key] = str(candidate.resolve())
                        break
            self.save_run(run)

        return {
            "archived": bool(moved_paths),
            "moved_paths": moved_paths,
        }

    def delete_legacy_batch(self, batch_id: str) -> Dict[str, Any]:
        imported = any((run.get("legacy_lookup") or {}).get("batch_id") == batch_id for run in self.list_runs())
        if imported:
            raise ValueError("This legacy batch has already been imported. Archive it instead of deleting it.")

        deleted_paths: List[str] = []
        for root_name, root in [("saved_jobs", self.saved_jobs_dir), ("queued_jobs", self.queued_jobs_dir)]:
            candidate = root / batch_id
            if candidate.exists():
                shutil.rmtree(candidate)
                deleted_paths.append(str(candidate))
        trace_candidate = self.trace_dir / batch_id
        if trace_candidate.exists():
            shutil.rmtree(trace_candidate)
            deleted_paths.append(str(trace_candidate))
        if not deleted_paths:
            raise ValueError("Legacy batch not found.")
        return {
            "deleted": True,
            "deleted_paths": deleted_paths,
        }

    def migrate_legacy_batch(
        self,
        batch_id: str,
        *,
        apply_changes: bool = False,
        archive_after: bool = False,
    ) -> Dict[str, Any]:
        migrated_runs: List[Dict[str, Any]] = []
        found_any = False
        for root_name, batch_dir, claim_file in self._iter_legacy_claim_files(batch_id=batch_id):
            found_any = True
            claim_data = _read_json(claim_file)
            claim_text = claim_data.get("text", "")
            if not claim_text:
                continue
            claim_record, _ = self.get_or_create_claim(
                claim_text,
                batch_tags=[batch_dir.name],
                metadata={"migrated": True},
            )
            existing = self.find_run_by_legacy(batch_dir.name, claim_file.stem)
            migrated_runs.append(
                {
                    "batch_id": batch_dir.name,
                    "claim_id": claim_file.stem,
                    "claim_key": claim_record["claim_key"],
                    "source_root": root_name,
                    "status": claim_data.get("status", "unknown"),
                    "already_imported": existing is not None,
                }
            )
            if not apply_changes or existing:
                continue
            self._persist_imported_legacy_run(
                batch_dir=batch_dir,
                claim_file=claim_file,
                root_name=root_name,
                claim_data=claim_data,
                claim_record=claim_record,
            )

        if not found_any:
            raise ValueError("Legacy batch not found.")

        archive_summary = {"archived": False, "moved_paths": []}
        if apply_changes and archive_after:
            archive_summary = self.archive_legacy_batch(batch_id)
        return {
            "apply_changes": apply_changes,
            "archive_after": archive_after,
            "runs": migrated_runs,
            "migrated_count": len(migrated_runs),
            "archive": archive_summary,
        }

    def migrate_legacy(self, *, apply_changes: bool = False) -> Dict[str, Any]:
        migrated_runs: List[Dict[str, Any]] = []
        batch_ids = [batch["batch_id"] for batch in self.list_legacy_batches()]
        for batch_id in batch_ids:
            result = self.migrate_legacy_batch(batch_id, apply_changes=apply_changes)
            migrated_runs.extend(result["runs"])
        return {
            "apply_changes": apply_changes,
            "runs": migrated_runs,
            "migrated_count": len(migrated_runs),
        }

    def _decorate_run(self, run_record: Dict[str, Any]) -> Dict[str, Any]:
        run_record = dict(run_record)
        report = run_record.get("report")
        run_record["report_available"] = isinstance(report, dict) and bool(report)
        run_record.setdefault("stop_after", "final_report")
        run_record.setdefault("completed_stage", None)
        run_record.setdefault("is_stage_checkpoint", False)
        run_record.setdefault("seed_from_run_id", None)
        run_record.setdefault("prompt_provenance", default_prompt_provenance())
        if not run_record.get("location"):
            claim_path = self.resolve_report_path(run_record)
            if claim_path:
                if str(claim_path).startswith(str(self.saved_jobs_dir)):
                    run_record["location"] = "saved_jobs"
                elif str(claim_path).startswith(str(self.queued_jobs_dir)):
                    run_record["location"] = "queued_jobs"
                elif str(claim_path).startswith(str(self.archive_dir / "saved_jobs")):
                    run_record["location"] = "archived_saved_jobs"
                elif str(claim_path).startswith(str(self.archive_dir / "queued_jobs")):
                    run_record["location"] = "archived_queued_jobs"
        return run_record
