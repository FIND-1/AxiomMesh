import unittest
from unittest.mock import patch

from backend.llm.aggregation import LLMExecutionCollector, aggregate_llm_executions, log_llm_run_summary
from backend.llm.contracts import LLMUsage
from backend.llm.telemetry import LLMExecutionRecord


def make_record(
    *,
    execution_id="exec-1",
    run_id="run-1",
    workflow_role="specialist",
    logical_model="qwen",
    provider="qwen",
    success=True,
    attempt_count=1,
    retried=False,
    latency_ms=10.0,
    usage=None,
):
    return LLMExecutionRecord(
        execution_id=execution_id,
        run_id=run_id,
        workflow_role=workflow_role,
        logical_model=logical_model,
        model_id=f"{logical_model}/model",
        provider=provider,
        provider_model_id=f"{provider}-model",
        success=success,
        attempt_count=attempt_count,
        retried=retried,
        latency_ms=latency_ms,
        usage=usage,
    )


class LLMExecutionAggregationTest(unittest.TestCase):
    def test_empty_collector_returns_zero_summary(self):
        summary = LLMExecutionCollector("run-empty").summary()

        self.assertEqual(summary["run_id"], "run-empty")
        self.assertEqual(summary["invocation_count"], 0)
        self.assertEqual(summary["attempt_count"], 0)
        self.assertEqual(summary["success_count"], 0)
        self.assertEqual(summary["failure_count"], 0)
        self.assertEqual(summary["by_role"], {})

    def test_simple_success_aggregates_usage(self):
        summary = aggregate_llm_executions(
            [
                make_record(
                    usage=LLMUsage(
                        input_tokens=10,
                        output_tokens=5,
                        total_tokens=15,
                        cached_tokens=2,
                        reasoning_tokens=1,
                    )
                )
            ],
            run_id="run-1",
        )

        self.assertEqual(summary["invocation_count"], 1)
        self.assertEqual(summary["attempt_count"], 1)
        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["failure_count"], 0)
        self.assertEqual(summary["confirmed_usage"]["input_tokens"]["known_sum"], 10)
        self.assertTrue(summary["confirmed_usage"]["input_tokens"]["is_complete"])

    def test_retry_success_counts_invocation_and_attempts_separately(self):
        summary = aggregate_llm_executions(
            [
                make_record(
                    attempt_count=2,
                    retried=True,
                    usage=LLMUsage(total_tokens=42),
                )
            ]
        )

        self.assertEqual(summary["invocation_count"], 1)
        self.assertEqual(summary["attempt_count"], 2)
        self.assertEqual(summary["retried_invocations"], 1)
        self.assertEqual(summary["retry_attempts"], 1)

    def test_failure_usage_is_unavailable_not_zero(self):
        summary = aggregate_llm_executions(
            [
                make_record(
                    success=False,
                    attempt_count=2,
                    retried=True,
                    usage=None,
                )
            ]
        )

        self.assertEqual(summary["failure_count"], 1)
        usage = summary["confirmed_usage"]["total_tokens"]
        self.assertEqual(usage["known_sum"], 0)
        self.assertTrue(usage["has_unknown"])
        self.assertFalse(usage["is_complete"])

    def test_mixed_usage_completeness_keeps_known_sum_and_unknown_flag(self):
        summary = aggregate_llm_executions(
            [
                make_record(execution_id="exec-1", usage=LLMUsage(input_tokens=10)),
                make_record(execution_id="exec-2", usage=LLMUsage(input_tokens=None)),
                make_record(execution_id="exec-3", usage=LLMUsage(input_tokens=20)),
            ]
        )

        usage = summary["confirmed_usage"]["input_tokens"]
        self.assertEqual(usage["known_sum"], 30)
        self.assertTrue(usage["has_unknown"])
        self.assertFalse(usage["is_complete"])

    def test_role_model_and_provider_grouping(self):
        summary = aggregate_llm_executions(
            [
                make_record(execution_id="exec-1", workflow_role="specialist", logical_model="qwen", provider="qwen"),
                make_record(execution_id="exec-2", workflow_role="specialist", logical_model="gemini", provider="gemini"),
                make_record(execution_id="exec-3", workflow_role="judge", logical_model="gemini", provider="gemini"),
                make_record(execution_id="exec-4", workflow_role="final", logical_model="gemini", provider="gemini"),
            ]
        )

        self.assertEqual(summary["by_role"]["specialist"]["invocation_count"], 2)
        self.assertEqual(summary["by_role"]["judge"]["invocation_count"], 1)
        self.assertEqual(summary["by_role"]["final"]["invocation_count"], 1)
        self.assertEqual(summary["by_logical_model"]["gemini"]["invocation_count"], 3)
        self.assertEqual(summary["by_provider"]["gemini"]["invocation_count"], 3)

    def test_latency_summary_is_invocation_latency_only(self):
        summary = aggregate_llm_executions(
            [
                make_record(execution_id="exec-1", latency_ms=10.0),
                make_record(execution_id="exec-2", latency_ms=15.0),
            ]
        )

        self.assertEqual(summary["latency"]["sum_invocation_latency_ms"], 25.0)
        self.assertEqual(summary["latency"]["avg_invocation_latency_ms"], 12.5)
        self.assertEqual(summary["latency"]["max_invocation_latency_ms"], 15.0)
        self.assertNotIn("run_latency_ms", summary["latency"])

    def test_run_scoped_collectors_are_isolated(self):
        collector_a = LLMExecutionCollector("run-a")
        collector_b = LLMExecutionCollector("run-b")
        collector_a.add(make_record(execution_id="exec-a", run_id="run-a"))
        collector_b.add(make_record(execution_id="exec-b", run_id="run-b"))

        self.assertEqual(collector_a.summary()["run_id"], "run-a")
        self.assertEqual(collector_b.summary()["run_id"], "run-b")
        self.assertEqual(collector_a.records()[0].execution_id, "exec-a")
        self.assertEqual(collector_b.records()[0].execution_id, "exec-b")

    def test_run_summary_uses_structured_logging(self):
        summary = LLMExecutionCollector("run-log").summary()

        with patch("backend.llm.aggregation.logger.info") as mock_info:
            log_llm_run_summary(summary)

        mock_info.assert_called_once_with("llm_run_summary %s", summary)


if __name__ == "__main__":
    unittest.main()