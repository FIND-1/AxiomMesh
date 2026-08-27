"""Incident input parsing helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Mapping, Optional


KNOWN_LOG_NAMES = ("error.log", "application.log", "system.log")


@dataclass(frozen=True)
class ParsedLogInput:
    """A parsed log attachment or pasted log section."""

    name: str
    content: str
    line_count: int
    non_empty_line_count: int


@dataclass(frozen=True)
class IncidentInput:
    """Normalized user incident input with optional logs."""

    user_query: str
    logs: Dict[str, ParsedLogInput]


def _normalize_log_name(value: str) -> Optional[str]:
    lower_value = value.strip().lower()
    for name in KNOWN_LOG_NAMES:
        if name in lower_value:
            return name
    return None


def _build_log_input(name: str, content: str) -> Optional[ParsedLogInput]:
    stripped = content.strip()
    if not stripped:
        return None

    lines = stripped.splitlines()
    return ParsedLogInput(
        name=name,
        content=stripped,
        line_count=len(lines),
        non_empty_line_count=sum(1 for line in lines if line.strip()),
    )


def _append_log(logs: Dict[str, str], name: str, content: str) -> None:
    stripped = content.strip()
    if not stripped:
        return

    if name in logs and logs[name].strip():
        logs[name] = f"{logs[name].rstrip()}\n\n{stripped}"
    else:
        logs[name] = stripped


def _extract_fenced_logs(content: str) -> Dict[str, str]:
    logs: Dict[str, str] = {}
    fence_pattern = re.compile(
        r"```(?P<label>[^\n`]*)\n(?P<body>.*?)```",
        flags=re.DOTALL,
    )

    for match in fence_pattern.finditer(content):
        name = _normalize_log_name(match.group("label"))
        if name:
            _append_log(logs, name, match.group("body"))

    return logs


def _match_log_section_header(line: str) -> Optional[str]:
    stripped = line.strip()
    if not stripped:
        return None

    plain_header_pattern = re.compile(
        r"^(?:#{1,6}\s*)?(?:\[(?P<bracket>[^\]]+)\]|(?P<plain>error\.log|application\.log|system\.log))\s*:?\s*$",
        flags=re.IGNORECASE,
    )
    marker_header_pattern = re.compile(
        r"^[=\-_*#\s]*(?P<name>error\.log|application\.log|system\.log)[=\-_*#:\s]*$",
        flags=re.IGNORECASE,
    )

    plain_match = plain_header_pattern.match(stripped)
    if plain_match:
        return _normalize_log_name(plain_match.group("bracket") or plain_match.group("plain") or "")

    marker_match = marker_header_pattern.match(stripped)
    if marker_match:
        return _normalize_log_name(marker_match.group("name"))

    return None


def _extract_section_logs(content: str) -> Dict[str, str]:
    logs: Dict[str, str] = {}
    current_name: Optional[str] = None
    current_lines: list[str] = []

    def flush_current() -> None:
        nonlocal current_name, current_lines
        if current_name:
            _append_log(logs, current_name, "\n".join(current_lines))
        current_name = None
        current_lines = []

    for line in content.splitlines():
        next_name = _match_log_section_header(line)
        if next_name:
            flush_current()
            current_name = next_name
            current_lines = []
            continue

        if current_name:
            current_lines.append(line)

    flush_current()
    return logs


def parse_incident_input(
    content: str,
    structured_logs: Optional[Mapping[str, str]] = None,
) -> IncidentInput:
    """Parse user content plus optional structured logs into incident input."""
    raw_logs: Dict[str, str] = {}

    for name, log_content in _extract_fenced_logs(content).items():
        _append_log(raw_logs, name, log_content)

    for name, log_content in _extract_section_logs(content).items():
        _append_log(raw_logs, name, log_content)

    if structured_logs:
        for raw_name, log_content in structured_logs.items():
            name = _normalize_log_name(raw_name)
            if name:
                _append_log(raw_logs, name, str(log_content))

    logs = {
        name: parsed
        for name, raw_content in raw_logs.items()
        if (parsed := _build_log_input(name, raw_content)) is not None
    }

    return IncidentInput(user_query=content.strip(), logs=logs)


def format_incident_context(incident_input: IncidentInput) -> str:
    """Format parsed logs for model prompts without hiding the original input."""
    if not incident_input.logs:
        return "No dedicated error.log, application.log, or system.log sections were detected."

    sections = ["Parsed log inputs:"]
    for name in KNOWN_LOG_NAMES:
        log = incident_input.logs.get(name)
        if not log:
            continue

        excerpt_lines = log.content.splitlines()[:80]
        excerpt = "\n".join(excerpt_lines)
        if len(log.content.splitlines()) > len(excerpt_lines):
            excerpt += "\n... [truncated for prompt context]"

        sections.append(
            f"\n{name} ({log.non_empty_line_count} non-empty lines):\n```text\n{excerpt}\n```"
        )

    return "\n".join(sections)


def summarize_incident_input(incident_input: IncidentInput) -> Dict[str, object]:
    """Return small metadata describing parsed user input."""
    return {
        "has_logs": bool(incident_input.logs),
        "logs": [
            {
                "name": name,
                "line_count": log.line_count,
                "non_empty_line_count": log.non_empty_line_count,
            }
            for name, log in incident_input.logs.items()
        ],
    }
