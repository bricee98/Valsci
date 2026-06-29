"""Processor liveness via a heartbeat file in the state directory.

The processor daemon touches the heartbeat on every poll loop; the web app
reads it to tell users whether queued claims will actually be picked up.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

HEARTBEAT_FILENAME = "processor_heartbeat.json"

# The processor touches the heartbeat every loop (~1s, throttled to every
# WRITE_INTERVAL_SECONDS). Anything older than STALE_AFTER_SECONDS means the
# daemon is down or wedged.
WRITE_INTERVAL_SECONDS = 5.0
STALE_AFTER_SECONDS = 20.0


def heartbeat_path(state_dir: str | Path) -> Path:
    return Path(state_dir) / HEARTBEAT_FILENAME


def write_heartbeat(
    state_dir: str | Path,
    *,
    now: Optional[float] = None,
    config_mtime: Optional[float] = None,
) -> None:
    path = heartbeat_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": os.getpid(),
        "timestamp": now if now is not None else time.time(),
        # mtime of env_vars.json the processor has loaded; lets the web app tell
        # whether the running processor reflects the latest saved settings.
        "config_mtime": config_mtime,
    }
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".heartbeat-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def read_heartbeat(state_dir: str | Path, *, now: Optional[float] = None) -> Dict[str, Any]:
    """Returns {"alive", "age_seconds", "pid", "config_mtime"}."""
    current = now if now is not None else time.time()
    path = heartbeat_path(state_dir)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        timestamp = float(payload.get("timestamp"))
    except (OSError, ValueError, TypeError):
        return {"alive": False, "age_seconds": None, "pid": None, "config_mtime": None}
    age = max(0.0, current - timestamp)
    return {
        "alive": age <= STALE_AFTER_SECONDS,
        "age_seconds": round(age, 1),
        "pid": payload.get("pid"),
        "config_mtime": payload.get("config_mtime"),
    }
