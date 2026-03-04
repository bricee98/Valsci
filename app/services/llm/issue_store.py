"""Durable issue persistence and helpers for report surfacing."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.services.llm.types import IssueRecord, utc_now_iso


class IssueStore:
    def __init__(self, root_dir: str = "saved_jobs", enabled: bool = True):
        self.root_dir = Path(root_dir)
        self.enabled = enabled
        self._locks: Dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    async def add_issue(
        self,
        *,
        batch_id: Optional[str],
        claim_id: Optional[str],
        severity: str,
        stage: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled or not batch_id or not claim_id:
            return None
        issue = IssueRecord(
            issue_id=str(uuid.uuid4()),
            timestamp=utc_now_iso(),
            severity=severity.upper(),
            stage=stage,
            message=message,
            details=details or {},
        ).to_dict()
        path = self.issue_file_path(batch_id, claim_id)
        await self._append_jsonl(path, issue)
        return issue

    async def read_all(self, batch_id: str, claim_id: str) -> List[Dict[str, Any]]:
        path = self.issue_file_path(batch_id, claim_id)
        if not path.exists():
            return []
        lock = await self._get_lock(str(path))
        async with lock:
            issues: List[Dict[str, Any]] = []
            text = path.read_text(encoding="utf-8")
            for line in text.splitlines():
                if not line.strip():
                    continue
                try:
                    issues.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            return issues

    def issue_file_path(self, batch_id: str, claim_id: str) -> Path:
        return self.root_dir / batch_id / "issues" / f"{claim_id}.jsonl"

    async def _append_jsonl(self, path: Path, record: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(record, ensure_ascii=True)
        lock = await self._get_lock(str(path))
        async with lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(serialized + "\n")

    async def _get_lock(self, key: str) -> asyncio.Lock:
        async with self._global_lock:
            if key not in self._locks:
                self._locks[key] = asyncio.Lock()
            return self._locks[key]
