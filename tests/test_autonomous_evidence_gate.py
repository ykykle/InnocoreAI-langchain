import unittest

from agents.autonomous import AutonomousResearchGraph


class EvidenceGateTest(unittest.TestCase):
    def test_hallucinated_answer_is_replaced_when_search_has_no_evidence(self):
        answer, blocked = AutonomousResearchGraph._apply_evidence_gate(
            "Invented Paper: an unreliable model-only answer", True, {}
        )
        self.assertTrue(blocked)
        self.assertIn("未获得可验证的论文结果", answer)
        self.assertNotIn("Invented Paper", answer)

    def test_answer_is_preserved_when_verified_paper_exists(self):
        answer, blocked = AutonomousResearchGraph._apply_evidence_gate(
            "grounded answer", True, {"2606.01899v1": {"title": "Verified"}}
        )
        self.assertFalse(blocked)
        self.assertEqual(answer, "grounded answer")


if __name__ == "__main__":
    unittest.main()
