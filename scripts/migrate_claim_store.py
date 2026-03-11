#!/usr/bin/env python
"""Dry-run or apply migration from legacy batch folders into the canonical claim store."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from app.services.claim_store import ClaimStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import legacy saved_jobs/queued_jobs data into the claim store.")
    parser.add_argument("--apply", action="store_true", help="Write imported claim/run records into the state dir.")
    parser.add_argument("--state-dir", default=None, help="Override state directory path.")
    parser.add_argument("--saved-jobs-dir", default=None, help="Override saved jobs directory path.")
    parser.add_argument("--queued-jobs-dir", default=None, help="Override queued jobs directory path.")
    parser.add_argument("--trace-dir", default=None, help="Override trace directory path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = ClaimStore(
        state_dir=args.state_dir,
        saved_jobs_dir=args.saved_jobs_dir,
        queued_jobs_dir=args.queued_jobs_dir,
        trace_dir=args.trace_dir,
    )
    payload = store.migrate_legacy(apply_changes=args.apply)
    sys.stdout.write(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
