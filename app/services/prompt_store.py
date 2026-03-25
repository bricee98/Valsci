"""Shared prompt loading utilities."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional


PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
_PLACEHOLDER_PATTERN = re.compile(r"\{\{([a-zA-Z0-9_]+)\}\}")
STAGE_PROMPT_FILES = {
    "query_generation": {
        "system": "query_generation_system",
        "user": "query_generation_user",
    },
    "paper_analysis": {
        "system": "paper_analysis_system",
        "user": "paper_analysis_user",
    },
    "venue_scoring": {
        "system": "venue_scoring_system",
        "user": "venue_scoring_user",
    },
    "final_report": {
        "system": "final_report_system",
        "user": "final_report_user",
    },
}


def prompt_path(prompt_name: str) -> Path:
    return PROMPTS_DIR / f"{prompt_name}.txt"


def sha256_text(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _modified_at(path: Path) -> Optional[str]:
    try:
        timestamp = path.stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def prompt_file_snapshot(prompt_name: str) -> Dict[str, Any]:
    path = prompt_path(prompt_name)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    contents = path.read_text(encoding="utf-8").strip()
    return {
        "prompt_name": prompt_name,
        "file_name": path.name,
        "sha256": sha256_text(contents),
        "modified_at": _modified_at(path),
    }


def stage_prompt_provenance(
    stage: str,
    *,
    rendered_system_prompt: Optional[str] = None,
    rendered_user_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    prompt_names = STAGE_PROMPT_FILES.get(stage) or {}
    system_name = prompt_names.get("system")
    user_name = prompt_names.get("user")
    system_snapshot = prompt_file_snapshot(system_name) if system_name else None
    user_snapshot = prompt_file_snapshot(user_name) if user_name else None

    rendered_prompt_hash = None
    if rendered_system_prompt is not None or rendered_user_prompt is not None:
        rendered_prompt_hash = sha256_text(
            "\n".join(
                [
                    str(rendered_system_prompt or ""),
                    str(rendered_user_prompt or ""),
                ]
            )
        )

    prompt_set_hash = sha256_text(
        "\n".join(
            [
                system_snapshot["sha256"] if system_snapshot else "",
                user_snapshot["sha256"] if user_snapshot else "",
            ]
        )
    )

    return {
        "stage": stage,
        "system_prompt": system_snapshot,
        "user_prompt": user_snapshot,
        "rendered_prompt_hash": rendered_prompt_hash,
        "prompt_set_hash": prompt_set_hash,
    }


def default_prompt_provenance() -> Dict[str, Any]:
    return {
        stage: stage_prompt_provenance(stage)
        for stage in STAGE_PROMPT_FILES
    }


@lru_cache(maxsize=128)
def load_prompt(prompt_name: str) -> str:
    """Load a prompt file from the repository prompts directory."""
    path = prompt_path(prompt_name)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def render_prompt(prompt_name: str, **values: Any) -> str:
    """Render a prompt using {{placeholder}} substitutions."""
    rendered = load_prompt(prompt_name)
    for key, value in values.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", str(value))

    unresolved = sorted(set(_PLACEHOLDER_PATTERN.findall(rendered)))
    if unresolved:
        missing = ", ".join(unresolved)
        raise ValueError(f"Prompt '{prompt_name}' has unresolved placeholders: {missing}")
    return rendered
