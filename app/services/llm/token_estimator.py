"""Token estimation utilities with optional tokenizer-backed support."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


class TokenEstimator:
    """Estimate token counts for chat messages and plain text."""

    def __init__(self, heuristic_chars_per_token: float = 4.0):
        self.heuristic_chars_per_token = heuristic_chars_per_token
        self._tiktoken = None
        self._encodings = {}
        try:
            import tiktoken  # type: ignore

            self._tiktoken = tiktoken
        except Exception:
            self._tiktoken = None

    def estimate_text_tokens(self, text: str, model_hint: Optional[str] = None) -> int:
        if not text:
            return 0
        encoding = self._get_encoding(model_hint)
        if encoding is not None:
            try:
                return len(encoding.encode(text))
            except Exception:
                pass
        return max(1, math.ceil(len(text) / self.heuristic_chars_per_token))

    def estimate_chat_tokens(self, messages: List[Dict[str, Any]], model_hint: Optional[str] = None) -> int:
        if not messages:
            return 0
        # OpenAI chat messages have per-message overhead; keep conservative defaults.
        message_overhead = 4
        reply_overhead = 2
        total = reply_overhead
        for message in messages:
            total += message_overhead
            total += self.estimate_text_tokens(str(message.get("role", "")), model_hint)
            content = message.get("content", "")
            total += self.estimate_text_tokens(self._content_to_text(content), model_hint)
            if "name" in message:
                total += self.estimate_text_tokens(str(message["name"]), model_hint)
        return max(total, 1)

    def _get_encoding(self, model_hint: Optional[str]):
        if self._tiktoken is None:
            return None
        key = model_hint or "default"
        if key in self._encodings:
            return self._encodings[key]
        try:
            if model_hint:
                encoding = self._tiktoken.encoding_for_model(model_hint)
            else:
                encoding = self._tiktoken.get_encoding("cl100k_base")
        except Exception:
            try:
                encoding = self._tiktoken.get_encoding("cl100k_base")
            except Exception:
                encoding = None
        self._encodings[key] = encoding
        return encoding

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        return str(content)

