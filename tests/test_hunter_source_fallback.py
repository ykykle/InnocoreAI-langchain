import asyncio
import unittest
from unittest.mock import patch

from agents.hunter import HunterAgent


class HunterFallbackTest(unittest.TestCase):
    def test_missing_ieee_key_keeps_arxiv_results(self):
        hunter = HunterAgent(llm=object())
        hunter.config.external_apis.ieee_api_key = None

        async def fake_arxiv(*_args):
            return [{
                "id": "paper-1", "title": "Agent systems", "abstract": "agent systems",
                "authors": [], "pdf_url": "https://example.test/paper.pdf", "source": "arxiv",
            }]

        async def fake_download(paper):
            return paper

        with patch.object(hunter, "_search_papers_from_arxiv", fake_arxiv), \
                patch.object(hunter, "_download_and_save_paper", fake_download):
            result = asyncio.run(hunter.run({
                "keywords": ["agent systems"], "sources": ["arxiv", "ieee"], "max_papers": 1,
            }))

        self.assertEqual(len(result["papers"]), 1)
        self.assertEqual(result["source_results"], {"arxiv": 1})
        self.assertIn("IEEE_API_KEY", result["source_errors"]["ieee"])
        self.assertTrue(result["partial_success"])


if __name__ == "__main__":
    unittest.main()
