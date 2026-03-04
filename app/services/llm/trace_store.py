"""Durable per-claim trace persistence in JSONL files."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class TraceStore:
    def __init__(self, root_dir: str = "saved_jobs", enabled: bool = True):
        self.root_dir = Path(root_dir)
        self.enabled = enabled
        self._locks: Dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    async def append(self, batch_id: Optional[str], claim_id: Optional[str], record: Dict[str, Any]) -> Optional[Path]:
        if not self.enabled or not batch_id or not claim_id:
            return None
        path = self.trace_file_path(batch_id, claim_id)
        await self._append_jsonl(path, record)
        return path

    async def read_all(self, batch_id: str, claim_id: str) -> List[Dict[str, Any]]:
        path = self.trace_file_path(batch_id, claim_id)
        if not path.exists():
            return []
        lock = await self._get_lock(str(path))
        async with lock:
            records: List[Dict[str, Any]] = []
            text = path.read_text(encoding="utf-8")
            for line in text.splitlines():
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            return records

    def trace_file_path(self, batch_id: str, claim_id: str) -> Path:
        return self.root_dir / batch_id / "traces" / f"{claim_id}.jsonl"

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
