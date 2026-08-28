import unittest

from backend import council


class CouncilImportCompatibilityTest(unittest.TestCase):
    def test_facade_exports_public_and_patch_sensitive_symbols(self):
        expected_symbols = [
            "SPECIALIST_BLUEPRINTS",
            "SPECIALIST_JSON_SCHEMAS",
            "JUDGE_JSON_SCHEMA",
            "RoleBlueprint",
            "_extract_json_object",
            "_normalize_specialist_payload",
            "_normalize_judge_payload",
            "_is_valid_judge_payload",
            "_build_agent_result",
            "stage1_collect_responses",
            "stage2_judge_deliberation",
            "stage3_synthesize_final",
            "generate_conversation_title",
            "build_evidence_store_from_results",
            "build_council_metadata",
            "build_empty_council_results",
            "run_full_council",
            "query_model",
            "EvidenceStore",
            "run_investigation_tools",
            "log_llm_run_summary",
        ]

        missing = [symbol for symbol in expected_symbols if not hasattr(council, symbol)]

        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
