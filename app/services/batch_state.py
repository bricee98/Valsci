"""Helpers for building a merged batch view across saved and queued jobs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


CLAIM_FILE_SUFFIX = ".txt"
IGNORED_FILENAMES = {"claims.txt"}


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def list_claim_files(batch_dir: Path) -> List[Path]:
    if not batch_dir.exists() or not batch_dir.is_dir():
        return []
    claim_files = [
        path
        for path in batch_dir.glob(f"*{CLAIM_FILE_SUFFIX}")
        if path.name not in IGNORED_FILENAMES
    ]
    return sorted(claim_files, key=lambda path: path.stem)


def list_batch_ids(*, saved_jobs_root: Path, queued_jobs_root: Optional[Path] = None) -> List[str]:
    batch_ids = set()
    for root in [saved_jobs_root, queued_jobs_root]:
        if root is None or not root.exists() or not root.is_dir():
            continue
        for path in root.iterdir():
            if path.is_dir():
                batch_ids.add(path.name)
    return sorted(batch_ids)


def _claim_entry_from_path(path: Path, *, location: str) -> Dict[str, Any]:
    claim_data = read_json(path)
    report = claim_data.get("report")
    stat_result = path.stat()
    review_type = claim_data.get("review_type", "regular")
    status = claim_data.get("status", "unknown")
    return {
        "claim_id": path.stem,
        "text": claim_data.get("text", ""),
        "status": status,
        "location": location,
        "source_file": str(path.resolve()),
        "claim_data": claim_data,
        "report": report if isinstance(report, dict) else {},
        "report_available": isinstance(report, dict) and bool(report),
        "rating": report.get("claimRating") if isinstance(report, dict) else None,
        "review_type": review_type,
        "modified_at": datetime.fromtimestamp(stat_result.st_mtime).isoformat(),
        "modified_at_epoch": stat_result.st_mtime,
        "is_active": location == "queued_jobs" or status != "processed",
        "is_complete": location == "saved_jobs" and status == "processed",
        "locations": [location],
    }


def build_batch_state(
    *,
    batch_id: str,
    saved_jobs_root: Path,
    queued_jobs_root: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    merged_claims: Dict[str, Dict[str, Any]] = {}
    errors: List[Dict[str, str]] = []

    location_roots = [
        ("saved_jobs", saved_jobs_root),
        ("queued_jobs", queued_jobs_root),
    ]
    for location, root in location_roots:
        if root is None:
            continue
        batch_dir = root / batch_id
        for claim_path in list_claim_files(batch_dir):
            try:
                entry = _claim_entry_from_path(claim_path, location=location)
            except Exception as exc:
                errors.append(
                    {
                        "claim_id": claim_path.stem,
                        "source_file": str(claim_path.resolve()),
                        "location": location,
                        "error": str(exc),
                    }
                )
                continue

            existing = merged_claims.get(entry["claim_id"])
            if existing is None:
                merged_claims[entry["claim_id"]] = entry
                continue

            # Queued claims are the live source of truth during overlap windows.
            if location == "queued_jobs":
                entry["locations"] = sorted(set(existing.get("locations", [])) | {location, existing["location"]})
                merged_claims[entry["claim_id"]] = entry
            else:
                existing["locations"] = sorted(set(existing.get("locations", [])) | {location})
                merged_claims[entry["claim_id"]] = existing

    claims = sorted(merged_claims.values(), key=lambda item: item["claim_id"])
    if not claims and not errors:
        return None

    counts_by_status: Dict[str, int] = {}
    counts_by_location = {"saved_jobs": 0, "queued_jobs": 0}
    processed_claims = 0
    oldest_timestamp = None
    newest_timestamp = None
    current_claim = None

    for claim in claims:
        status = claim.get("status", "unknown") or "unknown"
        counts_by_status[status] = counts_by_status.get(status, 0) + 1

        location = claim.get("location")
        if location in counts_by_location:
            counts_by_location[location] += 1
        else:
            counts_by_location[location] = counts_by_location.get(location, 0) + 1

        modified_at_epoch = claim.get("modified_at_epoch")
        if oldest_timestamp is None or modified_at_epoch < oldest_timestamp:
            oldest_timestamp = modified_at_epoch
        if newest_timestamp is None or modified_at_epoch > newest_timestamp:
            newest_timestamp = modified_at_epoch

        if claim.get("is_complete"):
            processed_claims += 1
        if claim.get("location") == "queued_jobs" and status != "processed":
            current_claim = claim

    total_claims = len(claims)
    has_active_claims = counts_by_location.get("queued_jobs", 0) > 0
    has_partial_resume = has_active_claims and counts_by_location.get("saved_jobs", 0) > 0

    if total_claims == 0:
        batch_status = "initializing"
    elif not has_active_claims and all(claim.get("status") == "processed" for claim in claims):
        batch_status = "completed"
    else:
        batch_status = "processing"

    return {
        "batch_id": batch_id,
        "status": batch_status,
        "total_claims": total_claims,
        "processed_claims": processed_claims,
        "counts_by_status": counts_by_status,
        "counts_by_location": counts_by_location,
        "has_active_claims": has_active_claims,
        "has_partial_resume": has_partial_resume,
        "current_claim_id": current_claim.get("claim_id") if current_claim else None,
        "timestamp": datetime.fromtimestamp(oldest_timestamp).isoformat() if oldest_timestamp is not None else None,
        "updated_at": datetime.fromtimestamp(newest_timestamp).isoformat() if newest_timestamp is not None else None,
        "claims": claims,
        "errors": errors,
    }
