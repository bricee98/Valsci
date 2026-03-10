"""Helpers for exporting one or more saved batches as JSON or Markdown."""

from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.services.batch_state import build_batch_state, list_claim_files, read_json


def read_jsonl(path: Path) -> Tuple[List[Dict[str, Any]], int]:
    records: List[Dict[str, Any]] = []
    invalid_lines = 0
    open_fn = gzip.open if path.suffix == ".gz" else open
    with open_fn(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                invalid_lines += 1
                continue
            if isinstance(value, dict):
                records.append(value)
            else:
                invalid_lines += 1
    return records, invalid_lines


def resolve_artifact_path(
    *,
    trace_root: Path,
    saved_jobs_root: Path,
    batch_id: str,
    artifact_dir: str,
    claim_id: str,
    extension: str,
) -> Optional[Path]:
    candidates = [
        trace_root / batch_id / artifact_dir / f"{claim_id}.{extension}",
        trace_root / batch_id / artifact_dir / f"{claim_id}.{extension}.gz",
    ]
    if trace_root != saved_jobs_root:
        candidates.extend(
            [
                saved_jobs_root / batch_id / artifact_dir / f"{claim_id}.{extension}",
                saved_jobs_root / batch_id / artifact_dir / f"{claim_id}.{extension}.gz",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def safe_title(value: Any, fallback: str) -> str:
    text = str(value).strip() if value is not None else ""
    return text or fallback


def build_claim_export(
    *,
    claim_entry: Dict[str, Any],
    batch_id: str,
    saved_jobs_root: Path,
    trace_root: Path,
    include_traces: bool,
    include_issues: bool,
) -> Dict[str, Any]:
    claim_id = claim_entry["claim_id"]
    claim_data = claim_entry["claim_data"]
    result: Dict[str, Any] = {
        "claim_id": claim_id,
        "source_file": claim_entry["source_file"],
        "claim_location": claim_entry.get("location"),
        "report_available": bool(claim_entry.get("report_available", False)),
        "text": claim_entry.get("text", ""),
        "status": claim_entry.get("status", ""),
        "claim_data": claim_data,
    }

    artifacts: Dict[str, Any] = {}
    if include_traces:
        trace_path = resolve_artifact_path(
            trace_root=trace_root,
            saved_jobs_root=saved_jobs_root,
            batch_id=batch_id,
            artifact_dir="traces",
            claim_id=claim_id,
            extension="jsonl",
        )
        if trace_path:
            trace_records, invalid_lines = read_jsonl(trace_path)
            artifacts["trace_path"] = str(trace_path.resolve())
            artifacts["trace_invalid_lines"] = invalid_lines
            artifacts["trace_records"] = trace_records
        else:
            artifacts["trace_path"] = None
            artifacts["trace_invalid_lines"] = 0
            artifacts["trace_records"] = []

    if include_issues:
        issue_path = resolve_artifact_path(
            trace_root=trace_root,
            saved_jobs_root=saved_jobs_root,
            batch_id=batch_id,
            artifact_dir="issues",
            claim_id=claim_id,
            extension="jsonl",
        )
        if issue_path:
            issue_records, invalid_lines = read_jsonl(issue_path)
            artifacts["issue_path"] = str(issue_path.resolve())
            artifacts["issue_invalid_lines"] = invalid_lines
            artifacts["issue_records"] = issue_records
        else:
            artifacts["issue_path"] = None
            artifacts["issue_invalid_lines"] = 0
            artifacts["issue_records"] = []

    if artifacts:
        result["artifacts"] = artifacts

    return result


def build_batch_export(
    *,
    batch_id: str,
    saved_jobs_root: Path,
    queued_jobs_root: Optional[Path],
    trace_root: Path,
    include_traces: bool,
    include_issues: bool,
) -> Dict[str, Any]:
    batch_state = build_batch_state(
        batch_id=batch_id,
        saved_jobs_root=saved_jobs_root,
        queued_jobs_root=queued_jobs_root,
    )
    batch_dir = saved_jobs_root / batch_id
    if not batch_dir.exists() and queued_jobs_root is not None:
        queued_batch_dir = queued_jobs_root / batch_id
        if queued_batch_dir.exists():
            batch_dir = queued_batch_dir
    claim_entries = (batch_state or {}).get("claims", [])

    claims: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = list((batch_state or {}).get("errors", []))
    for claim_entry in claim_entries:
        try:
            claims.append(
                build_claim_export(
                    claim_entry=claim_entry,
                    batch_id=batch_id,
                    saved_jobs_root=saved_jobs_root,
                    trace_root=trace_root,
                    include_traces=include_traces,
                    include_issues=include_issues,
                )
            )
        except Exception as exc:
            errors.append(
                {
                    "claim_id": claim_entry.get("claim_id", "unknown"),
                    "source_file": claim_entry.get("source_file", ""),
                    "error": str(exc),
                }
            )

    return {
        "batch_id": batch_id,
        "batch_dir": str(batch_dir.resolve()),
        "batch_status": (batch_state or {}).get("status", "initializing"),
        "claim_count": len(claims),
        "claims": claims,
        "errors": errors,
    }


def markdown_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def json_code_block(value: Any) -> str:
    return "```json\n" + json.dumps(value, indent=2, ensure_ascii=True) + "\n```"


def markdown_list(lines: Iterable[str]) -> List[str]:
    return [f"- {line}" for line in lines]


def render_usage_lines(usage: Dict[str, Any]) -> List[str]:
    if not usage:
        return ["- None"]
    return markdown_list(
        [
            f"Input Tokens: {usage.get('input_tokens', 0)}",
            f"Output Tokens: {usage.get('output_tokens', 0)}",
            f"Total Tokens: {usage.get('total_tokens', 0)}",
            f"Estimated Cost USD: {usage.get('cost_usd', 0)}",
            f"Estimated Token Counts: {markdown_bool(usage.get('is_estimated', False))}",
        ]
    )


def render_named_papers(items: List[Dict[str, Any]], reason_key: Optional[str] = None) -> List[str]:
    if not items:
        return ["- None"]
    lines: List[str] = []
    for item in items:
        title = safe_title(item.get("title"), "Untitled Paper")
        link = item.get("link") or item.get("url") or ""
        reason = item.get(reason_key) if reason_key else None
        suffix = f" | {reason_key}: {reason}" if reason else ""
        if link:
            lines.append(f"- {title} | {link}{suffix}")
        else:
            lines.append(f"- {title}{suffix}")
    return lines


def render_processed_papers(items: List[Dict[str, Any]]) -> List[str]:
    if not items:
        return ["- None"]
    lines: List[str] = []
    for item in items:
        paper = item.get("paper", {})
        title = safe_title(paper.get("title"), "Untitled Paper")
        corpus_id = paper.get("corpusId", "N/A")
        relevance = item.get("relevance", "N/A")
        score = item.get("score", "N/A")
        lines.append(
            f"- {title} | corpusId={corpus_id} | relevance={relevance} | score={score}"
        )
    return lines


def render_queries(queries: List[str]) -> List[str]:
    if not queries:
        return ["- None"]
    return [f"- {query}" for query in queries]


def build_markdown(export_data: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Valsci Batch Export")
    lines.append("")
    lines.extend(
        markdown_list(
            [
                f"Exported At: {export_data['exported_at']}",
                f"Saved Jobs Dir: {export_data['saved_jobs_dir']}",
                f"Trace Dir: {export_data['trace_dir']}",
                f"Batches Requested: {', '.join(export_data['batch_ids'])}",
            ]
        )
    )
    lines.append("")

    for batch in export_data["batches"]:
        lines.append(f"## Batch {batch['batch_id']}")
        lines.append("")
        lines.extend(
            markdown_list(
                [
                    f"Claim Count: {batch['claim_count']}",
                    f"Batch Dir: {batch['batch_dir']}",
                    f"Read Errors: {len(batch.get('errors', []))}",
                ]
            )
        )
        lines.append("")

        if batch.get("errors"):
            lines.append("### Read Errors")
            lines.append("")
            for error in batch["errors"]:
                lines.append(
                    f"- {error.get('claim_id', 'unknown')}: {error.get('error', 'unknown error')}"
                )
            lines.append("")

        for claim in batch["claims"]:
            claim_data = claim.get("claim_data", {})
            report = claim_data.get("report", {}) or {}
            artifacts = claim.get("artifacts", {}) or {}

            lines.append(f"### Claim {claim['claim_id']}")
            lines.append("")

            lines.append("#### Input")
            lines.append("")
            lines.extend(
                markdown_list(
                    [
                        f"Status: {claim.get('status', '')}",
                        f"Claim Location: {claim.get('claim_location', '')}",
                        f"Report Available: {markdown_bool(claim.get('report_available', False))}",
                        f"Source File: {claim.get('source_file', '')}",
                    ]
                )
            )
            lines.append("")
            lines.append(claim.get("text", "").strip() or "(empty)")
            lines.append("")

            lines.append("#### Config")
            lines.append("")
            lines.append("Search Config:")
            lines.append(json_code_block(claim_data.get("search_config", {})))
            lines.append("")
            lines.append("Bibliometric Config:")
            lines.append(json_code_block(claim_data.get("bibliometric_config", {})))
            lines.append("")
            lines.append("Model Overrides:")
            lines.append(json_code_block(claim_data.get("model_overrides", {})))
            lines.append("")

            lines.append("#### Intermediate Results")
            lines.append("")
            lines.extend(
                markdown_list(
                    [
                        f"Raw Paper Count: {len(claim_data.get('raw_papers', []))}",
                        f"Processed Paper Count: {len(claim_data.get('processed_papers', []))}",
                        f"Non-Relevant Paper Count: {len(claim_data.get('non_relevant_papers', []))}",
                        f"Inaccessible Paper Count: {len(claim_data.get('inaccessible_papers', []))}",
                    ]
                )
            )
            lines.append("")
            lines.append("Search Queries:")
            lines.extend(render_queries(claim_data.get("semantic_scholar_queries", [])))
            lines.append("")
            lines.append("Candidate Papers:")
            lines.extend(
                render_named_papers(
                    [
                        {
                            "title": item.get("title"),
                            "url": item.get("url"),
                        }
                        for item in claim_data.get("raw_papers", [])
                    ]
                )
            )
            lines.append("")
            lines.append("Processed Papers:")
            lines.extend(render_processed_papers(claim_data.get("processed_papers", [])))
            lines.append("")
            lines.append("Non-Relevant Papers:")
            lines.extend(
                render_named_papers(
                    [
                        {
                            "title": item.get("paper", {}).get("title"),
                            "url": item.get("paper", {}).get("url"),
                            "explanation": item.get("explanation"),
                        }
                        for item in claim_data.get("non_relevant_papers", [])
                    ],
                    reason_key="explanation",
                )
            )
            lines.append("")
            lines.append("Inaccessible Papers:")
            lines.extend(
                render_named_papers(
                    claim_data.get("inaccessible_papers", []),
                    reason_key="access_reason",
                )
            )
            lines.append("")

            lines.append("#### Final Report")
            lines.append("")
            lines.extend(
                markdown_list(
                    [
                        f"Claim Rating: {report.get('claimRating', 'N/A')}",
                        f"Relevant Papers In Report: {len(report.get('relevantPapers', []))}",
                        f"Non-Relevant Papers In Report: {len(report.get('nonRelevantPapers', []))}",
                        f"Inaccessible Papers In Report: {len(report.get('inaccessiblePapers', []))}",
                    ]
                )
            )
            lines.append("")
            lines.append("Explanation:")
            lines.append("")
            lines.append(report.get("explanation", "No explanation available"))
            lines.append("")

            final_reasoning = report.get("finalReasoning")
            if final_reasoning:
                lines.append("Final Reasoning:")
                lines.append("")
                lines.append(final_reasoning)
                lines.append("")

            lines.append("Usage Summary:")
            lines.extend(render_usage_lines(report.get("usage_summary") or report.get("usage_stats") or {}))
            lines.append("")

            usage_by_stage = report.get("usage_by_stage", {})
            lines.append("Usage By Stage:")
            if usage_by_stage:
                for stage, stats in usage_by_stage.items():
                    lines.append(
                        f"- {stage}: in={stats.get('input_tokens', 0)} "
                        f"out={stats.get('output_tokens', 0)} "
                        f"total={stats.get('total_tokens', 0)} "
                        f"cost_usd={stats.get('cost_usd', 0)}"
                    )
            else:
                lines.append("- None")
            lines.append("")

            lines.append("Report Search Queries:")
            lines.extend(render_queries(report.get("searchQueries", [])))
            lines.append("")

            lines.append("Report Relevant Papers:")
            lines.extend(render_named_papers(report.get("relevantPapers", [])))
            lines.append("")

            lines.append("Issues:")
            report_issues = report.get("issues", [])
            if report_issues:
                for issue in report_issues:
                    lines.append(
                        f"- [{issue.get('severity', 'INFO')}] "
                        f"{issue.get('stage', 'system')}: {issue.get('message', '')}"
                    )
            else:
                lines.append("- None")
            lines.append("")

            debug_trace = report.get("debug_trace", {})
            lines.append("Debug Trace Summary:")
            if debug_trace:
                lines.append(json_code_block(debug_trace.get("summary", {})))
            else:
                lines.append("- None")
            lines.append("")

            if artifacts:
                lines.append("#### Artifacts")
                lines.append("")
                if "issue_records" in artifacts:
                    lines.extend(
                        markdown_list(
                            [
                                f"Issue Path: {artifacts.get('issue_path')}",
                                f"Issue Record Count: {len(artifacts.get('issue_records', []))}",
                                f"Issue Invalid Lines: {artifacts.get('issue_invalid_lines', 0)}",
                            ]
                        )
                    )
                    lines.append("")
                if "trace_records" in artifacts:
                    lines.extend(
                        markdown_list(
                            [
                                f"Trace Path: {artifacts.get('trace_path')}",
                                f"Trace Record Count: {len(artifacts.get('trace_records', []))}",
                                f"Trace Invalid Lines: {artifacts.get('trace_invalid_lines', 0)}",
                            ]
                        )
                    )
                    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_export_document(
    *,
    batch_ids: List[str],
    saved_jobs_root: Path,
    queued_jobs_root: Optional[Path],
    trace_root: Path,
    include_traces: bool,
    include_issues: bool,
) -> Dict[str, Any]:
    batches: List[Dict[str, Any]] = []
    missing_batches: List[str] = []
    for batch_id in batch_ids:
        batch_state = build_batch_state(
            batch_id=batch_id,
            saved_jobs_root=saved_jobs_root,
            queued_jobs_root=queued_jobs_root,
        )
        if batch_state is None or batch_state["total_claims"] == 0:
            missing_batches.append(batch_id)
            continue
        batches.append(
            build_batch_export(
                batch_id=batch_id,
                saved_jobs_root=saved_jobs_root,
                queued_jobs_root=queued_jobs_root,
                trace_root=trace_root,
                include_traces=include_traces,
                include_issues=include_issues,
            )
        )

    return {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "saved_jobs_dir": str(saved_jobs_root.resolve()),
        "trace_dir": str(trace_root.resolve()),
        "batch_ids": batch_ids,
        "missing_batches": missing_batches,
        "batches": batches,
    }
