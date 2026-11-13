from pathlib import Path
import tempfile
import unittest

from starter.src.chunking import by_paragraph, fixed_size
from starter.src.evaluation import evaluate_query, summarize
from starter.src.ingestion import is_content_page, normalize_text
from starter.src.models import DocumentPage
from starter.src.run import DATA_DIR, _resolve_pdf_sources


class IngestionTests(unittest.TestCase):
    def test_normalize_preserves_paragraph_boundaries_and_normalizes_unicode(self):
        normalized = normalize_text("Cafe\u0301\t  test\r\n\r\nnext")
        self.assertEqual(normalized, "Café test\n\nnext")

    def test_content_page_excludes_bibliography(self):
        content = DocumentPage("doc", "source.pdf", 1, "x" * 200, {"section": "Chapter"})
        references = DocumentPage(
            "doc", "source.pdf", 2, "x" * 200, {"section": "Tài liệu tham khảo"}
        )
        self.assertTrue(is_content_page(content))
        self.assertFalse(is_content_page(references))


class CorpusSourceTests(unittest.TestCase):
    def test_explicit_pdf_paths_resolve_to_distinct_document_sources(self):
        with tempfile.NamedTemporaryFile(
            dir=DATA_DIR, prefix="additional-source-", suffix=".pdf", delete=False
        ) as temporary_pdf:
            additional_pdf = Path(temporary_pdf.name)

        try:
            sources = _resolve_pdf_sources(
                ["data/Report Data.pdf", f"data/{additional_pdf.name}"]
            )
        finally:
            additional_pdf.unlink()

        self.assertEqual(
            [source["document_id"] for source in sources],
            ["report-instacart", additional_pdf.stem.lower()],
        )
        self.assertEqual(
            [source["source"] for source in sources],
            [
                "starter/data/Report Data.pdf",
                f"starter/data/{additional_pdf.name}",
            ],
        )


class ChunkingTests(unittest.TestCase):
    def setUp(self):
        self.page = DocumentPage(
            "doc",
            "source.pdf",
            7,
            "abcdefghij",
            {"section": "Chapter 1"},
        )

    def test_fixed_size_overlap_and_traceability(self):
        chunks = fixed_size(self.page, size=6, overlap=2)
        self.assertEqual([chunk.text for chunk in chunks], ["abcdef", "efghij"])
        self.assertEqual(
            [(chunk.metadata["start_offset"], chunk.metadata["end_offset"]) for chunk in chunks],
            [(0, 6), (4, 10)],
        )
        self.assertTrue(all(chunk.page == 7 for chunk in chunks))

    def test_fixed_size_rejects_invalid_overlap(self):
        with self.assertRaises(ValueError):
            fixed_size(self.page, size=6, overlap=6)

    def test_paragraph_chunks_never_exceed_limit_and_keep_source_offsets(self):
        page = DocumentPage("doc", "source.pdf", 1, "first line\nsecond line\n\nthird paragraph")
        chunks = by_paragraph(page, max_size=16)
        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text), 16)
            start = chunk.metadata["start_offset"]
            end = chunk.metadata["end_offset"]
            self.assertEqual(page.text[start:end], chunk.text)


class EvaluationTests(unittest.TestCase):
    def test_retrieval_requires_matching_source_page_and_evidence(self):
        question = {
            "id": "q1",
            "split": "dev",
            "question": "What?",
            "expected_answer": "Answer",
            "gold_source_id": "doc",
            "gold_page": 2,
            "evidence_quote": "verified evidence",
        }
        hits = [
            {"text": "verified evidence", "metadata": {"document_id": "other", "page": 2}},
            {"text": "unrelated text", "metadata": {"document_id": "doc", "page": 2}},
            {"text": "verified evidence", "metadata": {"document_id": "doc", "page": 2}},
        ]
        trace = evaluate_query(question, hits)
        self.assertFalse(trace["hit_at_1"])
        self.assertTrue(trace["evidence_hit_at_3"])

    def test_summary_separates_strategy_and_split(self):
        traces = [
            {
                "strategy": "A",
                "split": "dev",
                "hit_at_1": True,
                "evidence_hit_at_3": True,
            },
            {
                "strategy": "A",
                "split": "test",
                "hit_at_1": False,
                "evidence_hit_at_3": True,
            },
        ]
        summary = summarize(traces)
        self.assertEqual(len(summary), 2)
        self.assertEqual(summary[0]["hit_at_1_count"], 1)
        self.assertEqual(summary[1]["evidence_hit_at_3_count"], 1)


if __name__ == "__main__":
    unittest.main()
