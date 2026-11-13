"""Run the three-strategy Day 06 ingestion and retrieval benchmark."""

import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .chunking import by_paragraph, fixed_size
from .embeddings import DEFAULT_MODEL, SentenceTransformerEmbedder
from .evaluation import evaluate_query, summarize, validate_questions
from .ingestion import is_content_page, load_pdf
from .models import Chunk, DocumentPage
from .store import ChromaVectorStore


ROOT = Path(__file__).resolve().parents[2]
STARTER = ROOT / "starter"
DATA_DIR = STARTER / "data"
QUESTIONS_PATH = ROOT / "benchmark" / "questions.json"
OUTPUT_DIR = ROOT / "results"
SELECTION_PATH = ROOT / "benchmark" / "selection.json"
MIN_CONTENT_PAGES = 30
STRATEGIES = ("A", "B", "C")


def _read_questions(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = data["questions"] if isinstance(data, dict) else data
    if not isinstance(questions, list):
        raise ValueError("Benchmark JSON must contain a list of questions.")
    return questions


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV to {path}.")
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _resolve_pdf_sources(
    requested_paths: list[str] | None,
) -> list[dict[str, str]]:
    candidates = (
        [Path(path) for path in requested_paths]
        if requested_paths
        else sorted(DATA_DIR.glob("*.pdf"))
    )
    if not candidates:
        raise FileNotFoundError(f"No PDF files found in {DATA_DIR}.")

    sources: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    data_root = DATA_DIR.resolve()
    for candidate in candidates:
        path = candidate if candidate.is_absolute() else STARTER / candidate
        path = path.resolve(strict=True)
        if path.suffix.casefold() != ".pdf":
            raise ValueError(f"Corpus source must be a PDF: {path}")
        try:
            relative = path.relative_to(data_root)
        except ValueError as exc:
            raise ValueError(
                f"PDF sources must be stored under {DATA_DIR}: {path}"
            ) from exc

        if path.name.casefold() == "report data.pdf":
            document_id = "report-instacart"
            title = "Phân tích hành vi mua sắm Instacart trên Big Data"
        else:
            document_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", path.stem.lower()).strip("-")
            title = path.stem
            if not document_id:
                raise ValueError(f"Could not create a document ID from {path.name!r}.")
        if document_id in seen_ids:
            raise ValueError(f"PDF filenames produce duplicate document ID {document_id!r}.")
        seen_ids.add(document_id)
        sources.append(
            {
                "document_id": document_id,
                "title": title,
                "path": str(path),
                "source": (Path("starter") / "data" / relative).as_posix(),
            }
        )
    return sources


def _write_manifest(
    raw_pages: list[DocumentPage],
    pages: list[DocumentPage],
    sources: list[dict[str, str]],
) -> None:
    useful_locations = {(page.document_id, page.page) for page in pages}
    documents = []
    for source in sources:
        document_id = source["document_id"]
        document_pages = [page for page in raw_pages if page.document_id == document_id]
        documents.append(
            {
                "source_id": document_id,
                "document_id": document_id,
                "title": source["title"],
                "source_path_or_url": source["source"],
                "source_url": None,
                "license_or_permission": (
                    "Chưa xác minh; cần xác nhận quyền sử dụng trước khi nộp."
                ),
                "accessed_at": (
                    "Chưa rõ ngày truy cập nguồn gốc; file được thêm vào workspace "
                    "ngày 2026-11-13."
                ),
                "snapshot_sha256": hashlib.sha256(
                    Path(source["path"]).read_bytes()
                ).hexdigest(),
                "physical_page_count": len(document_pages),
                "useful_content_page_count": sum(
                    (page.document_id, page.page) in useful_locations
                    for page in document_pages
                ),
                "pages": [
                    {
                        "page": page.page,
                        "section": page.metadata.get("section"),
                        "character_count": len(page.text),
                        "counted_as_content": (
                            page.document_id,
                            page.page,
                        ) in useful_locations,
                    }
                    for page in document_pages
                ],
            }
        )
    _write_json(
        ROOT / "data" / "manifest.json",
        {
            "documents": documents,
            "content_page_rule": (
                "Trang có ít nhất 150 ký tự trích xuất; loại trang tài liệu tham khảo."
            ),
            "physical_page_count": len(raw_pages),
            "useful_content_page_count": len(pages),
        },
    )


def _strategy_chunks(
    pages: list[DocumentPage],
) -> tuple[dict[str, list[Chunk]], dict[str, float]]:
    started = time.perf_counter()
    fixed_chunks = [
        chunk
        for page in pages
        for chunk in fixed_size(page, size=300, overlap=40)
    ]
    fixed_seconds = time.perf_counter() - started
    started = time.perf_counter()
    paragraph_chunks = [
        chunk for page in pages for chunk in by_paragraph(page, max_size=300)
    ]
    paragraph_seconds = time.perf_counter() - started
    return (
        {"A": fixed_chunks, "B": paragraph_chunks, "C": paragraph_chunks},
        {"A": fixed_seconds, "B": paragraph_seconds, "C": paragraph_seconds},
    )


def _chunk_stats(chunks: list[Chunk], token_counts: list[int]) -> dict[str, Any]:
    lengths = [len(chunk.text) for chunk in chunks]
    return {
        "chunk_count": len(chunks),
        "total_characters": sum(lengths),
        "min_characters": min(lengths) if lengths else 0,
        "median_characters": statistics.median(lengths) if lengths else 0,
        "max_characters": max(lengths) if lengths else 0,
        "total_tokens": sum(token_counts),
        "median_tokens": statistics.median(token_counts) if token_counts else 0,
    }


def _write_traces(path: Path, traces: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for trace in traces:
            output.write(json.dumps(trace, ensure_ascii=False) + "\n")


def _select_from_dev(
    summaries: list[dict[str, Any]], chunk_counts: dict[str, int],
    corpus_requirement_met: bool,
) -> dict[str, Any]:
    dev_rows = [row for row in summaries if row["split"] == "dev"]
    if {row["strategy"] for row in dev_rows} != set(STRATEGIES):
        raise ValueError("Dev metrics must include all three strategies before selection.")

    selected = max(
        dev_rows,
        key=lambda row: (
            row["evidence_hit_at_3"],
            row["hit_at_1"],
            -chunk_counts[row["strategy"]],
        ),
    )
    return {
        "selected_strategy": selected["strategy"],
        "selected_after": "dev",
        "selection_rule": [
            "Highest dev Evidence Hit@3",
            "Then highest dev Hit@1",
            "Then fewer chunks",
        ],
        "selection_metrics": selected,
        "corpus_requirement_met": corpus_requirement_met,
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _validate_test_gate() -> None:
    if not SELECTION_PATH.exists():
        raise ValueError(
            "Run the dev split first; test evaluation is gated until a dev selection exists."
        )
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    if selection.get("selected_after") != "dev":
        raise ValueError("The chosen strategy must be fixed from dev before test.")
    if selection.get("selected_strategy") not in STRATEGIES:
        raise ValueError("selection.json must name one of A, B, or C.")


def _coherence_anchor(page: DocumentPage) -> int:
    fallback = 0
    for line in re.finditer(r"\S[^\n]*", page.text):
        text = line.group().strip()
        if not text:
            continue
        if not fallback:
            fallback = line.start()
        normalized = text.casefold()
        if (
            normalized.startswith("phân tích hành vi mua sắm instacart")
            or normalized == f"trang {page.page}".casefold()
            or normalized.startswith(("chương ", "kết luận", "tài liệu tham khảo", "hình "))
        ):
            continue
        if len(text) >= 60:
            return line.start()
    return fallback


def _write_manual_audits(
    pages: list[DocumentPage],
    strategy_chunks: dict[str, list[Chunk]],
    test_traces: list[dict[str, Any]],
) -> None:
    source_by_document = {
        page.document_id: page.source
        for page in pages
    }
    content_pages = sorted(pages, key=lambda page: (page.document_id, page.page))
    sample_count = min(10, len(content_pages))
    if sample_count == 0:
        raise ValueError("No content pages are available for the coherence audit.")
    if sample_count == 1:
        sample_pages = content_pages
    else:
        indexes = {
            round(index * (len(content_pages) - 1) / (sample_count - 1))
            for index in range(sample_count)
        }
        sample_pages = [content_pages[index] for index in sorted(indexes)]

    coherence_rows: list[dict[str, Any]] = []
    for page in sample_pages:
        anchor = _coherence_anchor(page)
        for strategy in STRATEGIES:
            covering = [
                chunk for chunk in strategy_chunks[strategy]
                if chunk.document_id == page.document_id
                and chunk.page == page.page
                and chunk.metadata["start_offset"] <= anchor
                < chunk.metadata["end_offset"]
            ]
            selected_chunk = covering[0] if covering else None
            coherence_rows.append(
                {
                    "strategy": strategy,
                    "document_id": page.document_id,
                    "page": page.page,
                    "source_anchor_preview": page.text[anchor : anchor + 160],
                    "chunk_id": selected_chunk.chunk_id if selected_chunk else "",
                    "chunk_text": selected_chunk.text if selected_chunk else "",
                    "score_0_to_2": "",
                    "reason": "",
                }
            )
    _write_csv(OUTPUT_DIR / "coherence_audit.csv", coherence_rows)

    metadata_rows: list[dict[str, Any]] = []
    grounding_rows: list[dict[str, Any]] = []
    test_by_strategy = {
        strategy: [
            trace for trace in test_traces
            if trace["strategy"] == strategy
        ]
        for strategy in STRATEGIES
    }
    for strategy, traces in test_by_strategy.items():
        for trace in traces:
            formatted_hits = []
            for rank, hit in enumerate(trace["hits"], start=1):
                metadata = hit.get("metadata") or {}
                filter_values_match: bool | str = (
                    all(
                        metadata.get(key) == value
                        for key, value in trace["filter"].items()
                    )
                    if trace.get("filter")
                    else "N/A"
                )
                metadata_rows.append(
                    {
                        "strategy": strategy,
                        "question_id": trace["question_id"],
                        "rank": rank,
                        "chunk_id": hit["chunk_id"],
                        "gold_source_id": trace["gold_source_id"],
                        "gold_page": trace["gold_page"],
                        "expected_source": source_by_document.get(
                            trace["gold_source_id"], ""
                        ),
                        "source": metadata.get("source", ""),
                        "page": metadata.get("page", ""),
                        "filter": json.dumps(
                            trace.get("filter"), ensure_ascii=False
                        ),
                        "filter_values_match": filter_values_match,
                        "metadata_correct_manual": "",
                        "notes": "",
                    }
                )
                formatted_hits.append(
                    {
                        "rank": rank,
                        "chunk_id": hit["chunk_id"],
                        "source": metadata.get("source", ""),
                        "page": metadata.get("page", ""),
                        "text": hit["text"],
                    }
                )
            grounding_rows.append(
                {
                    "strategy": strategy,
                    "question_id": trace["question_id"],
                    "question": trace["question"],
                    "top3_evidence": json.dumps(formatted_hits, ensure_ascii=False),
                    "score_0_to_2": "",
                    "reason": "",
                }
            )
    if metadata_rows:
        _write_csv(OUTPUT_DIR / "metadata_audit.csv", metadata_rows)
    if grounding_rows:
        _write_csv(OUTPUT_DIR / "grounding_audit.csv", grounding_rows)


def run(
    phase: str,
    allow_short_corpus: bool,
    model_name: str,
    requested_pdf_paths: list[str] | None = None,
) -> None:
    if phase == "test":
        _validate_test_gate()

    sources = _resolve_pdf_sources(requested_pdf_paths)
    extraction_started = time.perf_counter()
    raw_pages = [
        page
        for source in sources
        for page in load_pdf(
            Path(source["path"]),
            source["document_id"],
            source["source"],
        )
    ]
    extraction_seconds = time.perf_counter() - extraction_started
    pages = [page for page in raw_pages if is_content_page(page)]
    if not pages:
        raise ValueError("The PDF contains no extractable content pages.")
    _write_manifest(raw_pages, pages, sources)
    corpus_requirement_met = len(pages) >= MIN_CONTENT_PAGES
    if not corpus_requirement_met and not allow_short_corpus:
        raise ValueError(
            f"Found {len(pages)} useful content pages; at least "
            f"{MIN_CONTENT_PAGES} are required. Add permitted source material, "
            "or pass --allow-short-corpus for explicitly preliminary development runs."
        )
    if not corpus_requirement_met:
        print(
            f"WARNING: running preliminary results with {len(pages)} content pages; "
            f"the {MIN_CONTENT_PAGES}-page requirement is not met.",
            file=sys.stderr,
        )

    questions = _read_questions(QUESTIONS_PATH)
    pages_by_location = {(page.document_id, page.page): page for page in raw_pages}
    validate_questions(questions, pages_by_location)
    for question in questions:
        gold_page = pages_by_location[(question["gold_source_id"], question["gold_page"])]
        for key, value in (question.get("filter") or {}).items():
            if gold_page.metadata.get(key) != value:
                raise ValueError(
                    f"Filter for {question['id']} does not match gold page metadata."
                )

    phase_questions = [question for question in questions if question["split"] == phase]
    if not phase_questions:
        raise ValueError(f"No {phase} questions are defined.")

    strategy_chunks, chunking_seconds = _strategy_chunks(pages)
    embedder = SentenceTransformerEmbedder(model_name)
    store = ChromaVectorStore(
        OUTPUT_DIR / "chroma",
        dimensions=embedder.dimensions,
    )
    store.create_schema()

    chunk_stats: dict[str, dict[str, Any]] = {}
    ingest_seconds: dict[str, float] = {}
    embedding_seconds: dict[str, float] = {}
    store_seconds: dict[str, float] = {}
    vectors_by_strategy: dict[str, list[list[float]]] = {}
    token_counts = {
        strategy: embedder.count_tokens(
            [chunk.text for chunk in strategy_chunks[strategy]]
        )
        for strategy in ("A", "B")
    }
    token_counts["C"] = token_counts["B"]
    for strategy in ("A", "B"):
        chunks = strategy_chunks[strategy]
        started = time.perf_counter()
        vectors = embedder.embed_documents([chunk.text for chunk in chunks])
        embedding_seconds[strategy] = time.perf_counter() - started
        store_started = time.perf_counter()
        store.upsert(strategy, chunks, vectors)
        store_seconds[strategy] = time.perf_counter() - store_started
        vectors_by_strategy[strategy] = vectors
        ingest_seconds[strategy] = (
            extraction_seconds
            + chunking_seconds[strategy]
            + embedding_seconds[strategy]
            + store_seconds[strategy]
        )
        chunk_stats[strategy] = _chunk_stats(chunks, token_counts[strategy])

    # C changes only retrieval filtering, so it reuses B's exact chunks and vectors.
    started = time.perf_counter()
    store.upsert("C", strategy_chunks["C"], vectors_by_strategy["B"])
    store_seconds["C"] = time.perf_counter() - started
    ingest_seconds["C"] = (
        extraction_seconds
        + chunking_seconds["C"]
        + embedding_seconds["B"]
        + store_seconds["C"]
    )
    chunk_stats["C"] = _chunk_stats(strategy_chunks["C"], token_counts["C"])

    query_embedding_started = time.perf_counter()
    query_vectors = {
        question["id"]: embedder.embed_query(question["question"])
        for question in phase_questions
    }
    query_embedding_seconds = time.perf_counter() - query_embedding_started
    traces: list[dict[str, Any]] = []
    retrieval_seconds = {strategy: 0.0 for strategy in STRATEGIES}
    for strategy in STRATEGIES:
        for question in phase_questions:
            filters = question.get("filter") if strategy == "C" else None
            retrieval_started = time.perf_counter()
            hits = store.search(
                strategy,
                query_vectors[question["id"]],
                top_k=3,
                filters=filters,
            )
            retrieval_seconds[strategy] += time.perf_counter() - retrieval_started
            trace = evaluate_query(question, hits)
            trace["strategy"] = strategy
            trace["corpus_requirement_met"] = corpus_requirement_met
            trace["manual_assessment"] = "pending"
            traces.append(trace)

    metrics = summarize(traces)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_traces(OUTPUT_DIR / f"{phase}_traces.jsonl", traces)
    _write_csv(OUTPUT_DIR / f"{phase}_metrics.csv", metrics)
    _write_json(
        OUTPUT_DIR / f"{phase}_ingestion_stats.json",
        {
            "run_at_utc": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "embedding_model": embedder.model_name,
            "embedding_model_revision": embedder.revision,
            "embedding_dimensions": embedder.dimensions,
            "embedding_max_sequence_length": embedder.max_sequence_length,
            "vector_store": "Chroma PersistentClient",
            "versions": {
                "python": sys.version.split()[0],
                "pypdf": version("pypdf"),
                "chromadb": version("chromadb"),
                "sentence-transformers": version("sentence-transformers"),
            },
            "distance_metric": "cosine",
            "corpus_files": [source["source"] for source in sources],
            "physical_pages": len(raw_pages),
            "useful_content_pages": len(pages),
            "required_content_pages": MIN_CONTENT_PAGES,
            "corpus_requirement_met": corpus_requirement_met,
            "pdf_extraction_seconds": extraction_seconds,
            "query_embedding_seconds": query_embedding_seconds,
            "strategies": {
                strategy: {
                    **chunk_stats[strategy],
                    "chunking": (
                        "fixed-size 300 chars, overlap 40"
                        if strategy == "A"
                        else "source-line groups, max 300 chars"
                    ),
                    "retrieval_filter": (
                        "question section metadata filter"
                        if strategy == "C"
                        else "none"
                    ),
                    "chunking_seconds": chunking_seconds[strategy],
                    "embedding_seconds": (
                        embedding_seconds.get(strategy, embedding_seconds["B"])
                    ),
                    "vector_store_seconds": store_seconds[strategy],
                    "ingestion_seconds": ingest_seconds[strategy],
                    "retrieval_seconds": retrieval_seconds[strategy],
                    "estimated_service_cost": 0.0,
                }
                for strategy in STRATEGIES
            },
        },
    )

    if phase == "dev":
        selection = _select_from_dev(
            metrics,
            {strategy: chunk_stats[strategy]["chunk_count"] for strategy in STRATEGIES},
            corpus_requirement_met,
        )
        _write_json(SELECTION_PATH, selection)
    else:
        _write_manual_audits(pages, strategy_chunks, traces)

    print(f"Completed {phase} retrieval run for {len(phase_questions)} questions.")
    print(f"Metrics: {OUTPUT_DIR / f'{phase}_metrics.csv'}")
    print(f"Traces: {OUTPUT_DIR / f'{phase}_traces.jsonl'}")
    if phase == "dev":
        print(f"Dev selection: {SELECTION_PATH}")
    else:
        print(f"Manual audit sheets: {OUTPUT_DIR}")
    store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("dev", "test"),
        default="dev",
        help="Run dev first; test is gated on a recorded dev-based selection.",
    )
    parser.add_argument(
        "--allow-short-corpus",
        action="store_true",
        help="Permit explicitly preliminary runs below the required 30 content pages.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Sentence Transformers model name; keep it unchanged across phases.",
    )
    parser.add_argument(
        "--pdf",
        action="append",
        dest="pdf_paths",
        help=(
            "PDF path relative to starter/ or absolute; repeat to select multiple PDFs. "
            "By default, all PDFs directly inside starter/data/ are ingested."
        ),
    )
    args = parser.parse_args()
    run(args.phase, args.allow_short_corpus, args.model, args.pdf_paths)


if __name__ == "__main__":
    main()
