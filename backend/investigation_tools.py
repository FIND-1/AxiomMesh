"""Tool abstractions for the Investigation Agent."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol

from .incident_input import IncidentInput, KNOWN_LOG_NAMES


TIMESTAMP_PATTERN = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"
)
SEVERITY_PATTERNS = {
    "fatal": re.compile(r"\b(fatal|critical|panic)\b", re.IGNORECASE),
    "error": re.compile(r"\b(error|exception|traceback|failed|failure)\b", re.IGNORECASE),
    "warning": re.compile(r"\b(warn|warning)\b", re.IGNORECASE),
}


@dataclass(frozen=True)
class InvestigationToolContext:
    """Runtime context available to investigation tools."""

    incident_input: IncidentInput


class InvestigationTool(Protocol):
    """Common interface for deterministic investigation tools."""

    name: str
    description: str

    def run(self, context: InvestigationToolContext) -> Dict[str, Any]:
        """Run the tool and return a JSON-serializable result."""


def _normalize_signature(line: str) -> str:
    without_timestamp = TIMESTAMP_PATTERN.sub("<timestamp>", line)
    without_hex = re.sub(r"\b0x[0-9a-f]+\b", "<hex>", without_timestamp, flags=re.IGNORECASE)
    without_numbers = re.sub(r"\b\d+\b", "<num>", without_hex)
    collapsed = re.sub(r"\s+", " ", without_numbers).strip()
    return collapsed[:220]


def _severity_counts(lines: List[str]) -> Dict[str, int]:
    return {
        name: sum(1 for line in lines if pattern.search(line))
        for name, pattern in SEVERITY_PATTERNS.items()
    }


def _top_signatures(lines: List[str], limit: int = 5) -> List[Dict[str, Any]]:
    counts: Dict[str, int] = {}

    for line in lines:
        if not SEVERITY_PATTERNS["error"].search(line) and not SEVERITY_PATTERNS["fatal"].search(line):
            continue

        signature = _normalize_signature(line)
        if not signature:
            continue
        counts[signature] = counts.get(signature, 0) + 1

    return [
        {"signature": signature, "count": count}
        for signature, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]
    ]


class LogInputSummaryTool:
    """Summarize parsed log inputs for the Investigation Agent."""

    name = "log_input_summary"
    description = (
        "Summarizes parsed error.log, application.log, and system.log input sections, "
        "including severity counts, timestamps, and repeated failure signatures."
    )

    def run(self, context: InvestigationToolContext) -> Dict[str, Any]:
        logs = context.incident_input.logs
        if not logs:
            return {
                "tool_name": self.name,
                "status": "skipped",
                "summary": "No supported log inputs were detected.",
                "findings": [],
                "source": {
                    "source_type": "TOOL",
                    "name": self.name,
                    "location": "",
                    "timestamp": "",
                },
            }

        findings: List[Dict[str, Any]] = []
        for name in KNOWN_LOG_NAMES:
            log = logs.get(name)
            if not log:
                continue

            lines = [line for line in log.content.splitlines() if line.strip()]
            timestamps = [
                match.group(0)
                for line in lines
                for match in TIMESTAMP_PATTERN.finditer(line)
            ]
            findings.append(
                {
                    "log_name": name,
                    "line_count": log.line_count,
                    "non_empty_line_count": log.non_empty_line_count,
                    "severity_counts": _severity_counts(lines),
                    "first_timestamp": timestamps[0] if timestamps else None,
                    "last_timestamp": timestamps[-1] if timestamps else None,
                    "top_error_signatures": _top_signatures(lines),
                }
            )

        return {
            "tool_name": self.name,
            "status": "completed",
            "summary": f"Analyzed {len(findings)} parsed log input(s).",
            "findings": findings,
            "source": {
                "source_type": "TOOL",
                "name": self.name,
                "location": "",
                "timestamp": "",
            },
        }


AVAILABLE_INVESTIGATION_TOOLS: List[InvestigationTool] = [
    LogInputSummaryTool(),
]


def investigation_tool_catalog() -> List[Dict[str, str]]:
    """Return available investigation tools for prompt context and metadata."""
    return [
        {
            "name": tool.name,
            "description": tool.description,
        }
        for tool in AVAILABLE_INVESTIGATION_TOOLS
    ]


def run_investigation_tools(context: InvestigationToolContext) -> List[Dict[str, Any]]:
    """Run all currently available deterministic investigation tools."""
    return [tool.run(context) for tool in AVAILABLE_INVESTIGATION_TOOLS]


def format_tool_results(tool_results: List[Dict[str, Any]]) -> str:
    """Format tool results for model prompts."""
    if not tool_results:
        return "No investigation tools were run."

    chunks = []
    for result in tool_results:
        chunks.append(
            f"Tool: {result.get('tool_name', 'unknown')}\n"
            f"Status: {result.get('status', 'unknown')}\n"
            f"Summary: {result.get('summary', '')}\n"
            f"Findings: {result.get('findings', [])}"
        )

    return "\n\n".join(chunks)
