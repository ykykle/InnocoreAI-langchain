import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from agents.miner import MinerAgent


class MinerIdentifierRoutingTest(unittest.TestCase):
    def test_arxiv_id_never_reaches_uuid_database_lookup(self):
        miner = MinerAgent(llm=object())
        paper = {
            "id": "2606.01899v1", "external_id": "2606.01899v1", "source": "arxiv",
            "title": "Test paper", "abstract": "Test abstract", "authors": [],
        }
        miner._resolve_paper_from_url = AsyncMock(return_value=paper)
        miner._parse_paper_content = AsyncMock(return_value={})
        miner._find_related_papers = AsyncMock(return_value=[])
        miner._perform_comparison_analysis = AsyncMock(return_value={})
        miner._create_analysis_report = AsyncMock(return_value={"summary": "ok"})

        with patch("agents.miner.db_manager.get_paper", new=AsyncMock()) as get_paper:
            result = asyncio.run(miner.run({"paper_id": "2606.01899v1"}))

        get_paper.assert_not_awaited()
        miner._resolve_paper_from_url.assert_awaited_once_with(
            "https://arxiv.org/abs/2606.01899v1"
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["paper_id"], "2606.01899v1")

    def test_uuid_detection(self):
        self.assertTrue(MinerAgent._is_uuid("123e4567-e89b-12d3-a456-426614174000"))
        self.assertFalse(MinerAgent._is_uuid("2606.01899v1"))


if __name__ == "__main__":
    unittest.main()
