import gc
import tempfile
import unittest
from pathlib import Path

from starter.src.models import Chunk
from starter.src.store import ChromaVectorStore


class ChromaVectorStoreTests(unittest.TestCase):
    def test_persistent_filtered_search_and_document_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ChromaVectorStore(Path(directory), dimensions=2)
            reopened = None
            try:
                first = Chunk(
                    "first",
                    "doc",
                    "source.pdf",
                    1,
                    "first page evidence",
                    {"section": "Chapter 1"},
                )
                store.upsert("A", [first], [[1.0, 0.0]])

                reopened = ChromaVectorStore(Path(directory), dimensions=2)
                hits = reopened.search(
                    "A",
                    [1.0, 0.0],
                    filters={"section": "Chapter 1"},
                )
                self.assertEqual([hit["chunk_id"] for hit in hits], ["first"])

                replacement = Chunk(
                    "replacement",
                    "doc",
                    "source.pdf",
                    2,
                    "replacement page evidence",
                    {"section": "Chapter 2"},
                )
                reopened.upsert("A", [replacement], [[0.0, 1.0]])
                final_hits = reopened.search("A", [0.0, 1.0])
                self.assertEqual(
                    [hit["chunk_id"] for hit in final_hits], ["replacement"]
                )
                self.assertEqual(
                    reopened.search(
                        "A",
                        [1.0, 0.0],
                        filters={"section": "Chapter 1"},
                    ),
                    [],
                )
            finally:
                if reopened is not None:
                    reopened.close()
                store.close()
                del reopened
                del store
                gc.collect()


if __name__ == "__main__":
    unittest.main()
