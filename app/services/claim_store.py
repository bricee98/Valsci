"""Canonical claim/run store layered on top of the file-backed processor."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from app.config.settings import Config
from app.services.llm.types import empty_usage
from app.services.stage_execution import (
    checkpoint_complete,
    continue_stage_for_run,
    normalize_stop_after,
    review_value_for_stage,
    stage_label,
    is_stage_checkpoint as stage_is_checkpoint,
)


WHITESPACE_RE = re.compile(r"\s+")


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
            claim_entry["runs"].append(self.hydrate_run(run))
        arena["claim_groups"] = sorted(
            claim_groups.values(),
            key=lambda item: (item.get("text", ""), item.get("claim_key", "")),
        )
        arena["current_stage"] = normalize_stop_after(arena.get("current_stage"))
        arena["stage_history"] = stage_history
        return arena

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

    def build_run_summary(self, run_record: Dict[str, Any]) -> Dict[str, Any]:
        report = run_record.get("report") or {}
        usage = report.get("usage_summary") or run_record.get("usage") or empty_usage()
        transport = run_record.get("transport") or {}
        provider_snapshot = run_record.get("provider_snapshot") or {}
        stop_after = normalize_stop_after(run_record.get("stop_after"))
        completed_stage = run_record.get("completed_stage")
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
            "effective_models": self.effective_models(run_record),
            "model_overrides": run_record.get("model_overrides") or {},
            "search_config": run_record.get("search_config") or {},
            "bibliometric_config": run_record.get("bibliometric_config") or {},
            "cost_estimate": run_record.get("cost_estimate") or {},
            "cost_confirmation": run_record.get("cost_confirmation") or {},
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
        artifact_paths = run_record.get("artifact_paths") or {}
        for key in ["saved_jobs_file", "queued_jobs_file"]:
            artifact_path = artifact_paths.get(key)
            if artifact_path:
                path = Path(artifact_path)
                if path.exists():
                    return path
        transport = run_record.get("transport") or {}
        batch_id = transport.get("batch_id")
        claim_id = transport.get("claim_id")
        if not batch_id or not claim_id:
            return None
        for root in [self.saved_jobs_dir, self.queued_jobs_dir]:
            candidate = root / batch_id / f"{claim_id}.txt"
            if candidate.exists():
                return candidate
        return None

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
            "provider_snapshot": run_record.get("provider_snapshot") or {},
            "model_overrides": run_record.get("model_overrides") or {},
            "search_config": run_record.get("search_config") or {},
            "bibliometric_config": run_record.get("bibliometric_config") or {},
            "cost_estimate": run_record.get("cost_estimate") or {},
            "cost_confirmation": run_record.get("cost_confirmation") or {},
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
        for candidate in [issue_dir / f"{claim_id}.jsonl", issue_dir / f"{claim_id}.jsonl.gz"]:
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

    def build_claim_detail(self, claim_key: str) -> Optional[Dict[str, Any]]:
        claim = self.get_claim(claim_key)
        if not claim:
            return None
        runs = [self.build_run_summary(run) for run in self.list_runs_for_claim(claim_key)]
        runs.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        return {
            "claim": claim,
            "runs": runs,
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

    def discover_legacy_batches(self) -> List[str]:
        migrated = set()
        for run in self.list_runs():
            legacy = run.get("legacy_lookup") or {}
            batch_id = legacy.get("batch_id")
            if batch_id:
                migrated.add(batch_id)

        discovered = set()
        for root in [self.saved_jobs_dir, self.queued_jobs_dir]:
            if not root.exists():
                continue
            for path in root.iterdir():
                if path.is_dir() and any(path.glob("*.txt")):
                    discovered.add(path.name)
        return sorted(batch_id for batch_id in discovered if batch_id not in migrated)

    def migration_status(self) -> Dict[str, Any]:
        pending = self.discover_legacy_batches()
        return {
            "state_dir": str(self.state_dir.resolve()),
            "pending_batches": pending,
            "pending_count": len(pending),
        }

    def migrate_legacy(self, *, apply_changes: bool = False) -> Dict[str, Any]:
        migrated_runs: List[Dict[str, Any]] = []
        for root_name, root in [("saved_jobs", self.saved_jobs_dir), ("queued_jobs", self.queued_jobs_dir)]:
            if not root.exists():
                continue
            for batch_dir in sorted(path for path in root.iterdir() if path.is_dir()):
                for claim_file in sorted(batch_dir.glob("*.txt")):
                    if claim_file.name == "claims.txt":
                        continue
                    claim_data = _read_json(claim_file)
                    claim_text = claim_data.get("text", "")
                    if not claim_text:
                        continue
                    claim_record, _ = self.get_or_create_claim(claim_text, batch_tags=[batch_dir.name], metadata={"migrated": True})
                    existing = self.find_run_by_legacy(batch_dir.name, claim_file.stem)
                    migrated_runs.append(
                        {
                            "batch_id": batch_dir.name,
                            "claim_id": claim_file.stem,
                            "claim_key": claim_record["claim_key"],
                            "source_root": root_name,
                            "status": claim_data.get("status", "unknown"),
                        }
                    )
                    if not apply_changes or existing:
                        continue
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
                    self.save_run(run_record)
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
        if not run_record.get("location"):
            claim_path = self.resolve_report_path(run_record)
            if claim_path:
                if str(claim_path).startswith(str(self.saved_jobs_dir)):
                    run_record["location"] = "saved_jobs"
                elif str(claim_path).startswith(str(self.queued_jobs_dir)):
                    run_record["location"] = "queued_jobs"
        return run_record
