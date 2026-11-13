"""Gold-evidence validation and retrieval metrics."""

import unicodedata
from collections import defaultdict
from typing import Any


def _canonical(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def validate_questions(
    questions: list[dict[str, Any]],
    pages_by_location: dict[tuple[str, int], Any],
) -> None:
    """Reject incomplete gold labels or evidence not present on its cited page."""
    if len(questions) < 20:
        raise ValueError(f"Benchmark needs at least 20 questions; found {len(questions)}.")

    ids = [question.get("id") for question in questions]
    if any(not question_id for question_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("Every benchmark question needs a unique, non-empty id.")

    splits = {split: 0 for split in ("dev", "test")}
    filtered_splits: set[str] = set()
    for question in questions:
        split = question.get("split")
        if split not in splits:
            raise ValueError(f"Question {question['id']} has invalid split {split!r}.")
        splits[split] += 1
        if question.get("filter"):
            filtered_splits.add(split)

        source_id = question.get("gold_source_id")
        page_number = question.get("gold_page")
        quote = question.get("evidence_quote")
        if not source_id or not isinstance(page_number, int) or not quote:
            raise ValueError(f"Question {question['id']} is missing gold evidence fields.")
        page = pages_by_location.get((source_id, page_number))
        if page is None:
            raise ValueError(
                f"Question {question['id']} cites missing page {source_id}:{page_number}."
            )
        if _canonical(quote) not in _canonical(page.text):
            raise ValueError(
                f"Evidence for {question['id']} is not present on page {page_number}."
            )
        for key, value in (question.get("filter") or {}).items():
            if page.metadata.get(key) != value:
                raise ValueError(
                    f"Filter {key!r} for {question['id']} does not match gold page metadata."
                )
        if not question.get("question") or not question.get("expected_answer"):
            raise ValueError(f"Question {question['id']} needs a question and answer.")

    if splits["dev"] < 15 or splits["test"] < 5:
        raise ValueError(
            "Benchmark needs at least 15 dev and 5 test questions; "
            f"found {splits['dev']} dev and {splits['test']} test."
        )
    if sum(bool(question.get("filter")) for question in questions) < 3:
        raise ValueError("Benchmark needs at least 3 metadata-filter questions.")
    if "test" not in filtered_splits:
        raise ValueError("At least one test question must exercise a metadata filter.")


def evaluate_query(
    question: dict[str, Any], hits: list[dict[str, Any]]
) -> dict[str, Any]:
    """Score only source/page matches containing the verified evidence quote."""
    gold_source = question["gold_source_id"]
    gold_pages = question.get("gold_pages", [question["gold_page"]])
    if question["gold_page"] not in gold_pages:
        gold_pages = [*gold_pages, question["gold_page"]]
    quote = _canonical(question["evidence_quote"])

    scored_hits: list[dict[str, Any]] = []
    for hit in hits:
        metadata = hit.get("metadata") or {}
        same_location = (
            metadata.get("document_id") == gold_source
            and metadata.get("page") in gold_pages
        )
        has_evidence = quote in _canonical(hit.get("text", ""))
        scored_hits.append(
            {
                **hit,
                "gold_location_match": same_location,
                "contains_gold_evidence": has_evidence,
                "evidence_hit": same_location and has_evidence,
            }
        )

    return {
        "question_id": question["id"],
        "split": question["split"],
        "question": question["question"],
        "expected_answer": question["expected_answer"],
        "filter": question.get("filter"),
        "gold_source_id": gold_source,
        "gold_page": question["gold_page"],
        "evidence_quote": question["evidence_quote"],
        "hits": scored_hits,
        "hit_at_1": bool(scored_hits and scored_hits[0]["evidence_hit"]),
        "evidence_hit_at_3": any(hit["evidence_hit"] for hit in scored_hits[:3]),
    }


def summarize(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate Hit@1 and Evidence Hit@3 by strategy and split."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        groups[(trace["strategy"], trace["split"])].append(trace)

    summaries: list[dict[str, Any]] = []
    for (strategy, split), rows in sorted(groups.items()):
        total = len(rows)
        hit1 = sum(bool(row["hit_at_1"]) for row in rows)
        hit3 = sum(bool(row["evidence_hit_at_3"]) for row in rows)
        summaries.append(
            {
                "strategy": strategy,
                "split": split,
                "questions": total,
                "hit_at_1_count": hit1,
                "hit_at_1": hit1 / total if total else 0.0,
                "evidence_hit_at_3_count": hit3,
                "evidence_hit_at_3": hit3 / total if total else 0.0,
            }
        )
    return summaries
