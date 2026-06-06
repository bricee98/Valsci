import sys
import time
import types
from pathlib import Path


sys.modules.setdefault("ijson", types.SimpleNamespace())
sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=object))

from app.services.data_manager import DataJobManager


# A tiny program that writes to BOTH stdout and stderr and exits nonzero.
_TINY_PROGRAM = (
    "import sys; "
    "print('hello-from-stdout'); "
    "sys.stdout.flush(); "
    "print('boom-on-stderr', file=sys.stderr); "
    "sys.stderr.flush(); "
    "sys.exit(3)"
)

_OK_PROGRAM = "print('all-good')"


def _run_job_and_wait(manager: DataJobManager, command, timeout=15):
    # Inject the job record directly so we exercise _run_job with an arbitrary
    # command (build_command only knows downloader operations).
    job_id = "testjob01"
    job = {
        "job_id": job_id,
        "operation": "test",
        "status": "queued",
        "created_at": "2026-06-05T00:00:00+00:00",
        "updated_at": "2026-06-05T00:00:00+00:00",
        "started_at": None,
        "ended_at": None,
        "exit_code": None,
        "error": None,
        "stderr_tail": [],
        "command": command,
        "command_display": " ".join(command),
        "options": {},
        "logs": [],
    }
    manager._write_job(job)
    manager._run_job(job_id)
    return manager.get_job(job_id)


def test_data_job_separates_streams_and_fails_on_nonzero_exit(tmp_path):
    manager = DataJobManager(state_dir=tmp_path / "state", project_root=tmp_path)
    job = _run_job_and_wait(manager, [sys.executable, "-c", _TINY_PROGRAM])

    assert job["status"] == "failed"
    assert job["exit_code"] == 3

    logs = job["logs"]
    streams = {entry.get("stream") for entry in logs}
    assert "stdout" in streams
    assert "stderr" in streams
    assert "system" in streams

    stdout_lines = [e["line"] for e in logs if e.get("stream") == "stdout"]
    stderr_lines = [e["line"] for e in logs if e.get("stream") == "stderr"]
    assert any("hello-from-stdout" in line for line in stdout_lines)
    assert any("boom-on-stderr" in line for line in stderr_lines)

    # stderr is preserved as a tail and summarized into the error.
    assert any("boom-on-stderr" in line for line in job["stderr_tail"])
    assert "boom-on-stderr" in (job.get("error") or "")

    # Backward-compatible `line` values remain present on every entry.
    assert all("line" in entry for entry in logs)


def test_data_job_success_on_zero_exit(tmp_path):
    manager = DataJobManager(state_dir=tmp_path / "state", project_root=tmp_path)
    job = _run_job_and_wait(manager, [sys.executable, "-c", _OK_PROGRAM])
    assert job["status"] == "success"
    assert job["exit_code"] == 0
    assert any(e.get("stream") == "stdout" and "all-good" in e["line"] for e in job["logs"])
    assert job["stderr_tail"] == []
