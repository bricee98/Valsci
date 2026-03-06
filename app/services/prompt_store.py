"""Shared prompt loading utilities."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any


PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
_PLACEHOLDER_PATTERN = re.compile(r"\{\{([a-zA-Z0-9_]+)\}\}")


@lru_cache(maxsize=128)
def load_prompt(prompt_name: str) -> str:
    """Load a prompt file from the repository prompts directory."""
    prompt_path = PROMPTS_DIR / f"{prompt_name}.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


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
