#!/usr/bin/env python
"""Export one or more saved Valsci batches as a single JSON or Markdown file."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from app.services.batch_export import build_export_document, build_markdown, list_claim_files, read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read processed batch data from saved_jobs and export one or more batches "
            "as a single aggregated JSON or Markdown document."
        )
    )
    parser.add_argument(
        "batch_ids",
        nargs="*",
        help=(
            "Optional batch IDs to export. If omitted, use --claim-regex to auto-select "
            "matching batches."
        ),
    )
    parser.add_argument(
        "--claim-regex",
        default=None,
        help=(
            "Python regex used to auto-select batches whose saved claim text matches. "
            "If batch IDs are provided, only those batches are scanned."
        ),
    )
    parser.add_argument(
        "--ignore-case",
        action="store_true",
        help="Apply case-insensitive matching to --claim-regex.",
    )
    parser.add_argument(
        "--saved-jobs-dir",
        default="saved_jobs",
        help="Directory containing processed batch folders. Default: saved_jobs",
    )
    parser.add_argument(
        "--trace-dir",
        default=None,
        help=(
            "Optional root directory for traces/issues. Defaults to the saved jobs "
            "directory. saved_jobs is always used as a fallback."
        ),
    )
    parser.add_argument(
        "--queued-jobs-dir",
        default=None,
        help=(
            "Optional directory containing in-progress batch folders. Defaults to a "
            "sibling queued_jobs directory when present."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Output format. Default: json",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write export to this file instead of stdout.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level. Default: 2",
    )
    parser.add_argument(
        "--include-traces",
        action="store_true",
        help="Include per-claim trace records from traces/<claim_id>.jsonl(.gz).",
    )
    parser.add_argument(
        "--include-issues",
        action="store_true",
        help="Include per-claim issue records from issues/<claim_id>.jsonl(.gz).",
    )
    parser.add_argument(
        "--include-artifacts",
        action="store_true",
        help="Shortcut for enabling both --include-traces and --include-issues.",
    )
    return parser.parse_args()


def select_batches_by_claim_regex(
    *,
    saved_jobs_root: Path,
    batch_ids: List[str],
    claim_regex: str,
    ignore_case: bool,
) -> List[str]:
    flags = re.IGNORECASE if ignore_case else 0
    pattern = re.compile(claim_regex, flags)

    if batch_ids:
        candidate_batch_ids = batch_ids
    else:
        candidate_batch_ids = sorted(
            path.name
            for path in saved_jobs_root.iterdir()
            if path.is_dir()
        )

    matched_batch_ids: List[str] = []
    for batch_id in candidate_batch_ids:
        batch_dir = saved_jobs_root / batch_id
        if not batch_dir.exists() or not batch_dir.is_dir():
            continue
        for claim_path in list_claim_files(batch_dir):
            claim_data = read_json(claim_path)
            claim_text = str(claim_data.get("text", ""))
            if pattern.search(claim_text):
                matched_batch_ids.append(batch_id)
                break
    return matched_batch_ids


def write_output(text: str, output_path: Path | None) -> None:
    if output_path is None:
        sys.stdout.write(text)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()

    saved_jobs_root = Path(args.saved_jobs_dir).resolve()
    trace_root = Path(args.trace_dir).resolve() if args.trace_dir else saved_jobs_root
    if args.queued_jobs_dir:
        queued_jobs_root = Path(args.queued_jobs_dir).resolve()
    else:
        sibling_queued = saved_jobs_root.parent / "queued_jobs"
        queued_jobs_root = sibling_queued.resolve() if sibling_queued.exists() else None
    output_path = Path(args.output).resolve() if args.output else None

    if not saved_jobs_root.exists():
        print(f"saved jobs directory does not exist: {saved_jobs_root}", file=sys.stderr)
        return 1

    batch_ids = []
    seen = set()
    for value in args.batch_ids:
        batch_id = value.strip()
        if not batch_id or batch_id in seen:
            continue
        seen.add(batch_id)
        batch_ids.append(batch_id)

    if args.claim_regex:
        try:
            batch_ids = select_batches_by_claim_regex(
                saved_jobs_root=saved_jobs_root,
                batch_ids=batch_ids,
                claim_regex=args.claim_regex,
                ignore_case=args.ignore_case,
            )
        except re.error as exc:
            print(f"invalid claim regex: {exc}", file=sys.stderr)
            return 1
        if not batch_ids:
            print("no batches matched the supplied claim regex", file=sys.stderr)
            return 1
    elif not batch_ids:
        print("provide one or more batch IDs or use --claim-regex", file=sys.stderr)
        return 1

    include_traces = args.include_traces or args.include_artifacts
    include_issues = args.include_issues or args.include_artifacts

    export_data = build_export_document(
        batch_ids=batch_ids,
        saved_jobs_root=saved_jobs_root,
        queued_jobs_root=queued_jobs_root,
        trace_root=trace_root,
        include_traces=include_traces,
        include_issues=include_issues,
    )

    if not export_data["batches"]:
        print("no requested batches were found", file=sys.stderr)
        return 1

    if export_data["missing_batches"]:
        print(
            "warning: missing batches: " + ", ".join(export_data["missing_batches"]),
            file=sys.stderr,
        )

    if args.format == "json":
        rendered = json.dumps(export_data, indent=args.indent, ensure_ascii=True) + "\n"
    else:
        rendered = build_markdown(export_data)

    write_output(rendered, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
