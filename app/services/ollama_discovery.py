"""Helpers for discovering local Ollama models on the web host."""

from __future__ import annotations

import re
import subprocess
from typing import Any, Dict, List


DISCOVERY_COMMANDS = (["ollama", "list"], ["ollama", "ls"])
TABLE_SPLIT_RE = re.compile(r"\s{2,}")


def parse_ollama_list_output(output: str) -> List[Dict[str, Any]]:
    lines = [line.rstrip() for line in str(output or "").splitlines() if line.strip()]
    if not lines:
        return []
    if lines[0].strip().lower().startswith("name"):
        lines = lines[1:]

    discovered: List[Dict[str, Any]] = []
    for line in lines:
        parts = TABLE_SPLIT_RE.split(line.strip())
        if not parts:
            continue
        model_name = str(parts[0]).strip()
        if not model_name:
            continue
        tag = None
        if ":" in model_name:
            _, tag = model_name.rsplit(":", 1)
        discovered.append(
            {
                "model_name": model_name,
                "label": model_name,
                "enabled": True,
                "context_window_tokens": 8192,
                "max_output_tokens_default": 1024,
                "supports_temperature": True,
                "supports_json_mode": True,
                "input_cost_per_million": 0.0,
                "output_cost_per_million": 0.0,
                "discovery_metadata": {
                    "model_id": parts[1] if len(parts) > 1 else None,
                    "size": parts[2] if len(parts) > 2 else None,
                    "modified": parts[3] if len(parts) > 3 else None,
                    "tag": tag,
                },
            }
        )
    return discovered


def discover_ollama_models(timeout_seconds: int = 30) -> List[Dict[str, Any]]:
    errors: List[str] = []
    for command in DISCOVERY_COMMANDS:
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            return parse_ollama_list_output(completed.stdout)
        except FileNotFoundError:
            raise RuntimeError("Ollama CLI was not found on the web host.") from None
        except subprocess.CalledProcessError as exc:
            stderr = str(exc.stderr or "").strip()
            if stderr:
                errors.append(stderr)
        except subprocess.TimeoutExpired:
            raise RuntimeError("Ollama model discovery timed out.") from None
    if errors:
        raise RuntimeError(errors[-1])
    return []
