"""Conversation title fallback helpers."""

from __future__ import annotations

import re


def _fallback_conversation_title(user_query: str) -> str:
    """Generate a local title when the title model is unavailable."""
    without_fences = re.sub(r"```.*?```", " ", user_query, flags=re.DOTALL)
    lines = [line.strip() for line in without_fences.splitlines()]

    for line in lines:
        if not line:
            continue
        if re.match(r"^(?:#{1,6}\s*)?(?:\[)?(?:error|application|system)\.log(?:\])?:?\s*$", line, re.IGNORECASE):
            continue

        candidate = re.sub(r"^[#>\-\s]+", "", line)
        candidate = re.sub(r"\s+", " ", candidate).strip(" ，。,.!！?？:：;；\"'")
        if len(candidate) < 2:
            continue
        if len(candidate) > 24:
            candidate = candidate[:21].rstrip() + "..."
        return candidate

    return "新会话"
