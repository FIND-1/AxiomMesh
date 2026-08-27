import unittest

from backend.incident_input import parse_incident_input
from backend.investigation_tools import InvestigationToolContext, run_investigation_tools


class IncidentInputParserTest(unittest.TestCase):
    def test_parse_fenced_log_blocks(self):
        incident = parse_incident_input(
            """API started returning 500s.

```error.log
2026-08-25 10:00:01 ERROR Redis timeout for request 42
2026-08-25 10:00:02 ERROR Redis timeout for request 43
```
"""
        )

        self.assertIn("error.log", incident.logs)
        self.assertEqual(incident.logs["error.log"].non_empty_line_count, 2)
        self.assertIn("Redis timeout", incident.logs["error.log"].content)

    def test_parse_named_sections_and_structured_logs(self):
        incident = parse_incident_input(
            """Checkout latency is high.

[application.log]
2026-08-25 10:01:00 WARN retry payment provider

system.log:
2026-08-25 10:01:03 kernel: disk pressure warning
""",
            {"error.log": "2026-08-25 10:01:05 FATAL worker crashed"},
        )

        self.assertEqual(set(incident.logs), {"application.log", "system.log", "error.log"})
        self.assertIn("worker crashed", incident.logs["error.log"].content)


class InvestigationToolsTest(unittest.TestCase):
    def test_log_input_summary_tool_counts_severity_and_signatures(self):
        incident = parse_incident_input(
            """Incident report.

```error.log
2026-08-25 10:00:01 ERROR Redis timeout for request 42
2026-08-25 10:00:02 ERROR Redis timeout for request 43
2026-08-25 10:00:03 WARN retrying request 43
```
"""
        )

        results = run_investigation_tools(InvestigationToolContext(incident))

        self.assertEqual(results[0]["tool_name"], "log_input_summary")
        self.assertEqual(results[0]["status"], "completed")
        finding = results[0]["findings"][0]
        self.assertEqual(finding["log_name"], "error.log")
        self.assertEqual(finding["severity_counts"]["error"], 2)
        self.assertEqual(finding["severity_counts"]["warning"], 1)
        self.assertEqual(finding["first_timestamp"], "2026-08-25 10:00:01")
        self.assertEqual(finding["top_error_signatures"][0]["count"], 2)


if __name__ == "__main__":
    unittest.main()
