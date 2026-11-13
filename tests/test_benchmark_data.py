import json
import unittest
from pathlib import Path

from starter.src.evaluation import validate_questions
from starter.src.ingestion import load_pdf


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkDataTests(unittest.TestCase):
    def test_questions_have_valid_page_evidence_and_filter_metadata(self):
        pages = load_pdf(
            ROOT / "starter" / "data" / "Report Data.pdf",
            "report-instacart",
            "starter/data/Report Data.pdf",
        )
        questions = json.loads(
            (ROOT / "benchmark" / "questions.json").read_text(encoding="utf-8")
        )["questions"]
        validate_questions(
            questions,
            {(page.document_id, page.page): page for page in pages},
        )

        self.assertEqual(len(questions), 20)
        self.assertEqual(sum(question["split"] == "dev" for question in questions), 15)
        self.assertEqual(sum(question["split"] == "test" for question in questions), 5)


if __name__ == "__main__":
    unittest.main()
