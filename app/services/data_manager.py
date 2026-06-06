"""Local Semantic Scholar data status and downloader job orchestration."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from app.config.settings import Config
from semantic_scholar.utils.binary_indexer import BinaryIndexer
from semantic_scholar.utils.downloader import (
    MANIFESTS_DIR,
    MiniCorpusManifestError,
    S2DatasetDownloader,
    configured_manifest_name,
    resolve_manifest_name,
    resolve_manifest_path,
)


DATASET_LABELS = {
    "papers": "Papers",
    "abstracts": "Abstracts",
    "authors": "Authors",
    "s2orc_v2": "S2ORC full text",
    "tldrs": "TLDRs",
}

DATASET_NOTES = {
    "papers": "Core paper metadata used for identifiers and bibliometrics.",
    "abstracts": "Abstract fallback when full text is not locally available.",
    "authors": "Author metadata used for H-index and citation enrichment.",
    "s2orc_v2": "Current Semantic Scholar full-text dataset. API name: s2orc_v2.",
    "tldrs": "Short summary fallback when richer text is unavailable.",
}

JOB_LOG_LIMIT = 600
STDERR_TAIL_LIMIT = 20
RELEASE_ID_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _format_bytes(size_bytes: int) -> str:
    value = float(max(0, size_bytes))
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if value < 1024 or unit == "PB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} PB"


def _json_file(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _dataset_file_stats(dataset_dir: Path) -> Dict[str, Any]:
    file_count = 0
    total_size = 0
    largest_file = None
    if dataset_dir.exists():
        for path in dataset_dir.glob("*.json"):
            if path.name == "metadata.json":
                continue
            file_count += 1
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            total_size += size
            if largest_file is None or size > largest_file["size_bytes"]:
                largest_file = {
                    "name": path.name,
                    "size_bytes": size,
                    "size_label": _format_bytes(size),
                }
    return {
        "file_count": file_count,
        "size_bytes": total_size,
        "size_label": _format_bytes(total_size),
        "largest_file": largest_file,
    }


def _default_downloader() -> S2DatasetDownloader:
    return S2DatasetDownloader()


def dataset_options() -> List[Dict[str, Any]]:
    downloader = _default_downloader()
    try:
        datasets = getattr(downloader, "supported_datasets", downloader.datasets_to_download)
        return [
            {
                "name": dataset,
                "label": DATASET_LABELS.get(dataset, dataset),
                "note": DATASET_NOTES.get(dataset, ""),
                "default": dataset in downloader.datasets_to_download,
            }
            for dataset in datasets
        ]
    finally:
        downloader.indexer.close()


def _file_sha256(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        return None


def _manifest_summary(path: Path) -> Dict[str, Any]:
    summary = {
        "path": str(path),
        "exists": path.exists(),
        "hash": None,
        "release_id": None,
        "topic_label": None,
        "mini_release_id": None,
        "claim_count": 0,
        "corpus_id_count": 0,
        "author_id_count": 0,
        "dataset_id_counts": {},
        "source_file_count": 0,
        "error": None,
    }
    if not path.exists():
        summary["error"] = f"Manifest not found: {path}"
        return summary
    summary["hash"] = _file_sha256(path)
    try:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError as exc:
            summary["error"] = f"Manifest is not valid JSON: {exc}"
            return summary
        if not isinstance(data, dict):
            summary["error"] = "Manifest must be a JSON object."
            return summary
        summary["release_id"] = data.get("release_id")
        summary["mini_release_id"] = data.get("mini_release_id")
        summary["topic_label"] = data.get("topic_label")
        if not summary["topic_label"] and "-mini-" in str(summary["mini_release_id"] or ""):
            summary["topic_label"] = str(summary["mini_release_id"]).split("-mini-", 1)[1]
        summary["claim_count"] = len(data.get("claims") or [])
        dataset_config = data.get("datasets") or {}
        if isinstance(dataset_config, dict):
            for dataset, config in dataset_config.items():
                if not isinstance(config, dict):
                    continue
                ids = config.get("author_ids") if dataset == "authors" else config.get("corpus_ids")
                if isinstance(ids, list):
                    summary["dataset_id_counts"][dataset] = len(ids)
        summary["corpus_id_count"] = sum(
            count
            for dataset, count in summary["dataset_id_counts"].items()
            if dataset != "authors"
        ) or len(data.get("corpus_ids") or [])
        summary["author_id_count"] = summary["dataset_id_counts"].get("authors", len(data.get("author_ids") or []))
        source_files = data.get("source_files") or data.get("datasets") or {}
        if isinstance(source_files, dict):
            total = 0
            for value in source_files.values():
                if isinstance(value, list):
                    total += len(value)
                elif isinstance(value, dict):
                    files = value.get("files") or value.get("source_files") or []
                    total += len(files) if isinstance(files, list) else 0
            summary["source_file_count"] = total
        if not str(summary["release_id"] or "").strip():
            summary["error"] = "Manifest must include release_id."
        elif not summary["dataset_id_counts"] and not summary["corpus_id_count"] and not summary["author_id_count"]:
            summary["error"] = "Manifest must include dataset-specific corpus_ids or author_ids."
    except Exception as exc:
        summary["error"] = str(exc)
    return summary


def _manifest_status() -> Dict[str, Any]:
    """Resolve the configured manifest and build a status summary for the Data page."""
    configured = str(getattr(Config, "SEMANTIC_SCHOLAR_MANIFEST", "") or "").strip() or configured_manifest_name()
    try:
        name = resolve_manifest_name()
        path = resolve_manifest_path()
    except MiniCorpusManifestError as exc:
        return {
            "manifest": configured,
            "manifests_dir": str(MANIFESTS_DIR),
            "path": None,
            "exists": False,
            "hash": None,
            "error": str(exc),
            "release_id": None,
            "topic_label": None,
            "mini_release_id": None,
            "claim_count": 0,
            "corpus_id_count": 0,
            "author_id_count": 0,
            "dataset_id_counts": {},
            "source_file_count": 0,
        }
    summary = _manifest_summary(path)
    summary["manifest"] = name
    summary["manifests_dir"] = str(path.parent)
    return summary


def _release_index_status(datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = 0
    present = 0
    healthy = 0
    missing = 0
    unhealthy = 0
    data_missing = 0
    affected_datasets: List[str] = []
    for dataset in datasets:
        dataset_has_issue = False
        if dataset.get("indices") and (not dataset.get("exists") or not dataset.get("file_count")):
            data_missing += 1
            dataset_has_issue = True
        for index in dataset.get("indices") or []:
            total += 1
            if not index.get("exists"):
                missing += 1
                dataset_has_issue = True
                continue
            present += 1
            if index.get("healthy"):
                healthy += 1
            else:
                unhealthy += 1
                dataset_has_issue = True
        if dataset_has_issue:
            affected_datasets.append(dataset["name"])

    if total == 0:
        state = "not_applicable"
        label = "No index mappings"
    elif present == 0:
        state = "missing"
        label = "Not indexed"
    elif missing or unhealthy or data_missing:
        state = "needs_attention"
        label = "Data/index attention needed"
    else:
        state = "ready"
        label = "Indexed"

    return {
        "state": state,
        "label": label,
        "total": total,
        "present": present,
        "healthy": healthy,
        "missing": missing,
        "unhealthy": unhealthy,
        "data_missing": data_missing,
        "affected_datasets": affected_datasets,
    }


def _mini_manifest_coverage(
    *,
    release_id: str,
    records_written: Dict[str, Any],
    manifest_summary: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    requested = manifest_summary.get("dataset_id_counts") or {}
    if not requested or manifest_summary.get("mini_release_id") != release_id:
        return None

    written = {
        dataset: _safe_int(records_written.get(dataset))
        for dataset in requested
    }
    missing = {
        dataset: max(_safe_int(expected) - written.get(dataset, 0), 0)
        for dataset, expected in requested.items()
        if written.get(dataset, 0) < _safe_int(expected)
    }
    state = "ready" if not missing else "stale"
    return {
        "state": state,
        "label": "Matches manifest" if state == "ready" else "Rebuild needed",
        "requested": {dataset: _safe_int(count) for dataset, count in requested.items()},
        "written": written,
        "missing": missing,
    }


def build_data_state() -> Dict[str, Any]:
    downloader = _default_downloader()
    indexer = BinaryIndexer(downloader.base_dir)
    try:
        base_dir = downloader.base_dir
        index_dir = downloader.index_dir
        manifest_summary = _manifest_status()
        metadata_releases = {
            path.name[: -len("_metadata.json")]
            for path in index_dir.glob("*_metadata.json")
            if path.name.endswith("_metadata.json")
        }
        release_dirs = {
            path.name
            for path in base_dir.iterdir()
            if path.is_dir() and RELEASE_ID_PATTERN.match(path.name)
        } if base_dir.exists() else set()
        release_ids = sorted(release_dirs | metadata_releases, reverse=True)

        releases = []
        for release_id in release_ids:
            release_dir = base_dir / release_id
            metadata = _json_file(release_dir / "mini_manifest_metadata.json")
            mini_summary = _json_file(release_dir / "mini_build_summary.json")
            stats = indexer.get_index_stats(release_id)

            dataset_names = set(downloader.datasets_to_download)
            if release_dir.exists():
                dataset_names.update(
                    child.name
                    for child in release_dir.iterdir()
                    if child.is_dir()
                    and child.name != "binary_indices"
                    and child.name in downloader.datasets_to_download
                )

            datasets = []
            release_size = 0
            dataset_order = downloader.datasets_to_download
            for dataset in sorted(dataset_names, key=lambda name: dataset_order.index(name) if name in dataset_order else 999):
                dataset_dir = release_dir / dataset
                file_stats = _dataset_file_stats(dataset_dir)
                release_size += file_stats["size_bytes"]
                expected_id_types = [
                    id_type
                    for _, id_type in downloader.dataset_id_fields.get(dataset, [])
                ]
                indices = []
                for id_type in expected_id_types:
                    key = f"{dataset}_{id_type}"
                    info = stats.get(key)
                    indices.append(
                        {
                            "id_type": id_type,
                            "exists": bool(info),
                            "entry_count": _safe_int((info or {}).get("entry_count")),
                            "healthy": bool((info or {}).get("healthy")),
                            "size_mb": float((info or {}).get("size_mb") or 0),
                            "created": (info or {}).get("created"),
                        }
                    )
                files_present = dataset_dir.exists() and file_stats["file_count"] > 0
                datasets.append(
                    {
                        "name": dataset,
                        "label": DATASET_LABELS.get(dataset, dataset),
                        "note": DATASET_NOTES.get(dataset, ""),
                        "path": str(dataset_dir),
                        "exists": dataset_dir.exists(),
                        "file_count": file_stats["file_count"],
                        "size_bytes": file_stats["size_bytes"],
                        "size_label": file_stats["size_label"],
                        "largest_file": file_stats["largest_file"],
                        "metadata_exists": (dataset_dir / "metadata.json").exists(),
                        "indices": indices,
                        "indexed": any(index["exists"] for index in indices),
                        "fully_indexed": files_present and bool(indices) and all(index["exists"] for index in indices),
                        "healthy": files_present and bool(indices) and all(index["healthy"] for index in indices if index["exists"]),
                    }
                )

            index_status = _release_index_status(datasets)
            records_written = mini_summary.get("records_written", {})
            releases.append(
                {
                    "release_id": release_id,
                    "path": str(release_dir),
                    "exists": release_dir.exists(),
                    "is_mini": "-mini-" in release_id or bool(metadata),
                    "source_release_id": metadata.get("source_release_id") or mini_summary.get("source_release_id"),
                    "topic_label": metadata.get("topic_label") or mini_summary.get("topic_label"),
                    "size_bytes": release_size,
                    "size_label": _format_bytes(release_size),
                    "datasets": datasets,
                    "index_status": index_status,
                    "records_written": records_written,
                    "manifest_coverage": _mini_manifest_coverage(
                        release_id=release_id,
                        records_written=records_written,
                        manifest_summary=manifest_summary,
                    ),
                }
            )

        latest_release = releases[0]["release_id"] if releases else None
        active_release = next((release for release in releases if release["release_id"] == latest_release), None)
        api_key = Config.SEMANTIC_SCHOLAR_API_KEY
        return {
            "base_dir": str(base_dir),
            "index_dir": str(index_dir),
            "api_key_present": bool(api_key and str(api_key).strip()),
            "latest_release": latest_release,
            "active_release": active_release,
            "releases": releases,
            "dataset_options": dataset_options(),
            "mini_manifest": manifest_summary,
        }
    finally:
        downloader.indexer.close()
        indexer.close()


def _validate_datasets(datasets: Iterable[str]) -> List[str]:
    allowed = {item["name"] for item in dataset_options()}
    normalized = []
    for dataset in datasets or []:
        value = str(dataset or "").strip()
        if not value:
            continue
        if value not in allowed:
            raise ValueError(f"Unsupported dataset: {value}")
        if value not in normalized:
            normalized.append(value)
    return normalized


class DataJobManager:
    def __init__(self, state_dir: Optional[Path] = None, project_root: Optional[Path] = None):
        self.project_root = Path(project_root or Config.PROJECT_ROOT).resolve()
        self.jobs_dir = Path(state_dir or Config.STATE_DIR) / "data_jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._processes: Dict[str, subprocess.Popen] = {}
        self._mark_interrupted_jobs()

    def _job_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def _read_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        path = self._job_path(job_id)
        if not path.exists():
            return None
        return _json_file(path)

    def _write_job(self, job: Dict[str, Any]) -> None:
        path = self._job_path(job["job_id"])
        tmp_path = path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(job, handle, indent=2, ensure_ascii=True)
        os.replace(tmp_path, path)

    def _update_job(self, job_id: str, **updates: Any) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._read_job(job_id)
            if not job:
                return None
            job.update(updates)
            job["updated_at"] = _utc_now()
            self._write_job(job)
            return job

    def _append_log(self, job_id: str, line: str, stream: str = "system") -> None:
        clean = line.rstrip("\r\n")
        if not clean:
            return
        with self._lock:
            job = self._read_job(job_id)
            if not job:
                return
            logs = list(job.get("logs") or [])
            # `line` is kept for backward compatibility; `stream` distinguishes
            # stdout / stderr / system (orchestration) output.
            logs.append({"timestamp": _utc_now(), "line": clean, "stream": stream})
            job["logs"] = logs[-JOB_LOG_LIMIT:]
            job["updated_at"] = _utc_now()
            self._write_job(job)

    def _mark_interrupted_jobs(self) -> None:
        for path in self.jobs_dir.glob("*.json"):
            job = _json_file(path)
            if job.get("status") in {"queued", "running", "cancel_requested"}:
                job["status"] = "interrupted"
                job["ended_at"] = _utc_now()
                job["updated_at"] = _utc_now()
                job.setdefault("logs", []).append(
                    {
                        "timestamp": _utc_now(),
                        "line": "Valsci restarted while this data job was active.",
                        "stream": "system",
                    }
                )
                self._write_job(job)

    def list_jobs(self, limit: int = 8) -> List[Dict[str, Any]]:
        jobs = [_json_file(path) for path in self.jobs_dir.glob("*.json")]
        jobs = [job for job in jobs if job.get("job_id")]
        jobs.sort(key=lambda job: job.get("created_at", ""), reverse=True)
        return jobs[:limit]

    def active_job(self) -> Optional[Dict[str, Any]]:
        for job in self.list_jobs(limit=20):
            if job.get("status") in {"queued", "running", "cancel_requested"}:
                return job
        return None

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._read_job(job_id)

    @staticmethod
    def _manifest_arg(payload: Dict[str, Any]) -> str:
        """Return a validated bare manifest filename from the payload, or '' for the default.

        Accepts the new ``manifest`` key (filename); tolerates a legacy ``manifest_path``
        value only when it is already a bare filename. Any path-like override is rejected.
        """
        raw = str(payload.get("manifest") or payload.get("manifest_path") or "").strip()
        if not raw:
            return ""
        # Validates filename-only; raises MiniCorpusManifestError (a ValueError) on paths.
        return resolve_manifest_name(raw)

    def build_command(self, operation: str, payload: Dict[str, Any]) -> List[str]:
        command = [sys.executable, "-u", "-m", "semantic_scholar.utils.downloader"]
        release = str(payload.get("release") or "").strip()

        datasets = _validate_datasets(payload.get("datasets") or [])
        manifest_name = self._manifest_arg(payload)
        if operation == "mini":
            command.append("--mini")
            if manifest_name:
                command.extend(["--mini-manifest", manifest_name])
            return command
        if operation == "full":
            if release and release != "latest":
                command.extend(["--release", release])
            if datasets:
                command.extend(["--datasets", *datasets])
            return command
        if operation == "index":
            if release and release != "latest":
                command.extend(["--release", release])
            command.append("--index-only")
            command.extend(datasets)
            return command
        if operation == "verify":
            if payload.get("mini"):
                command.append("--mini")
                if manifest_name:
                    command.extend(["--mini-manifest", manifest_name])
            elif release and release != "latest":
                command.extend(["--release", release])
            command.append("--verify")
            if datasets and not payload.get("mini"):
                command.extend(["--datasets", *datasets])
            return command
        raise ValueError(f"Unsupported data operation: {operation}")

    def start_job(self, operation: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            active = self.active_job()
            if active:
                raise RuntimeError(f"Data job already running: {active['job_id']}")
            command = self.build_command(operation, payload)
            job_id = uuid.uuid4().hex[:12]
            job = {
                "job_id": job_id,
                "operation": operation,
                "status": "queued",
                "created_at": _utc_now(),
                "updated_at": _utc_now(),
                "started_at": None,
                "ended_at": None,
                "exit_code": None,
                "error": None,
                "stderr_tail": [],
                "command": command,
                "command_display": " ".join(command),
                "options": {
                    "datasets": payload.get("datasets") or [],
                    "release": payload.get("release") or "latest",
                    "manifest": self._manifest_arg(payload),
                },
                "logs": [
                    {"timestamp": _utc_now(), "line": f"Queued data job: {operation}", "stream": "system"}
                ],
            }
            self._write_job(job)

        thread = threading.Thread(target=self._run_job, args=(job_id,), daemon=True)
        thread.start()
        return job

    def _run_job(self, job_id: str) -> None:
        job = self._update_job(job_id, status="running", started_at=_utc_now())
        if not job:
            return
        self._append_log(job_id, f"Command: {job['command_display']}", stream="system")
        process = None
        stderr_lines: List[str] = []
        stderr_lock = threading.Lock()
        try:
            env = dict(os.environ)
            env.setdefault("PYTHONUNBUFFERED", "1")
            env.setdefault("PYTHONIOENCODING", "utf-8")
            process = subprocess.Popen(
                job["command"],
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )
            with self._lock:
                self._processes[job_id] = process

            def pump(pipe, stream: str) -> None:
                try:
                    for line in pipe:
                        self._append_log(job_id, line, stream=stream)
                        if stream == "stderr":
                            text = line.rstrip("\r\n")
                            if text:
                                with stderr_lock:
                                    stderr_lines.append(text)
                finally:
                    try:
                        pipe.close()
                    except Exception:
                        pass

            readers: List[threading.Thread] = []
            for pipe, stream in ((process.stdout, "stdout"), (process.stderr, "stderr")):
                if pipe is None:
                    continue
                reader = threading.Thread(target=pump, args=(pipe, stream), daemon=True)
                reader.start()
                readers.append(reader)

            exit_code = process.wait()
            for reader in readers:
                reader.join()

            with stderr_lock:
                tail = stderr_lines[-STDERR_TAIL_LIMIT:]

            latest = self._read_job(job_id) or {}
            if latest.get("status") == "cancel_requested":
                status = "cancelled"
            else:
                status = "success" if exit_code == 0 else "failed"

            updates: Dict[str, Any] = {
                "status": status,
                "exit_code": exit_code,
                "ended_at": _utc_now(),
                "stderr_tail": tail,
            }
            if status == "failed":
                updates["error"] = (
                    " ".join(tail[-3:]) if tail else f"Process exited with code {exit_code}."
                )
            self._update_job(job_id, **updates)
            self._append_log(
                job_id, f"Data job {status} with exit code {exit_code}.", stream="system"
            )
        except Exception as exc:
            self._update_job(job_id, status="failed", ended_at=_utc_now(), error=str(exc))
            self._append_log(job_id, f"Data job failed: {exc}", stream="system")
        finally:
            with self._lock:
                self._processes.pop(job_id, None)

    def cancel_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._read_job(job_id)
            if not job:
                return None
            if job.get("status") not in {"queued", "running", "cancel_requested"}:
                return job
            job["status"] = "cancel_requested"
            job["updated_at"] = _utc_now()
            self._write_job(job)
            process = self._processes.get(job_id)
            if process and process.poll() is None:
                process.terminate()
        self._append_log(job_id, "Cancellation requested.")
        return self._read_job(job_id)


_DATA_JOB_MANAGER: Optional[DataJobManager] = None
_DATA_JOB_MANAGER_LOCK = threading.Lock()


def data_job_manager() -> DataJobManager:
    global _DATA_JOB_MANAGER
    with _DATA_JOB_MANAGER_LOCK:
        if _DATA_JOB_MANAGER is None:
            _DATA_JOB_MANAGER = DataJobManager()
        return _DATA_JOB_MANAGER
