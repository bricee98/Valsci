import os

from app.services.processor_heartbeat import (
    STALE_AFTER_SECONDS,
    heartbeat_path,
    read_heartbeat,
    write_heartbeat,
)


def test_missing_heartbeat_reads_as_down(tmp_path):
    result = read_heartbeat(tmp_path)
    assert result == {"alive": False, "age_seconds": None, "pid": None, "config_mtime": None}


def test_fresh_heartbeat_reads_as_alive(tmp_path):
    write_heartbeat(tmp_path, now=1000.0)
    result = read_heartbeat(tmp_path, now=1003.0)
    assert result["alive"] is True
    assert result["age_seconds"] == 3.0
    assert result["pid"] == os.getpid()


def test_heartbeat_round_trips_config_mtime(tmp_path):
    write_heartbeat(tmp_path, now=1000.0, config_mtime=1699999999.5)
    result = read_heartbeat(tmp_path, now=1001.0)
    assert result["config_mtime"] == 1699999999.5


def test_stale_heartbeat_reads_as_down(tmp_path):
    write_heartbeat(tmp_path, now=1000.0)
    result = read_heartbeat(tmp_path, now=1000.0 + STALE_AFTER_SECONDS + 1)
    assert result["alive"] is False
    assert result["age_seconds"] == STALE_AFTER_SECONDS + 1


def test_corrupt_heartbeat_reads_as_down(tmp_path):
    heartbeat_path(tmp_path).write_text("not json", encoding="utf-8")
    result = read_heartbeat(tmp_path)
    assert result == {"alive": False, "age_seconds": None, "pid": None, "config_mtime": None}


def test_write_creates_state_dir_and_overwrites(tmp_path):
    target = tmp_path / "nested" / "state"
    write_heartbeat(target, now=50.0)
    write_heartbeat(target, now=60.0)
    result = read_heartbeat(target, now=61.0)
    assert result["alive"] is True
    assert result["age_seconds"] == 1.0
    # No stray temp files left behind by the atomic write.
    assert [p.name for p in target.iterdir()] == ["processor_heartbeat.json"]
