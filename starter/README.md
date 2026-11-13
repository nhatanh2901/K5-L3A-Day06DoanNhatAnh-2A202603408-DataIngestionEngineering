# Starter · local PDF ingestion benchmark

This starter runs locally with **Chroma PersistentClient** and a real multilingual
Sentence Transformers model. Docker and API keys are not required. The first run
downloads Python dependencies and the embedding model; allow network access and
enough disk space.

## Setup

Use Python 3.11–3.14 from the repository root:

```powershell
cd starter
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Put permitted PDF source material in `starter/data/`. By default, every PDF
directly inside that folder is ingested. To run on a selected set, pass each
source with a repeated `--pdf` option (paths are relative to `starter/`):

```powershell
python -m src.run --phase dev --pdf "data/Report Data.pdf" --pdf "data/Another Source.pdf"
```

The pipeline records per-document provenance and page counts in
`data/manifest.json` at the repository root. Add benchmark questions with
verified evidence from any additional sources to `benchmark/questions.json`.
Unknown rights/access information is marked as unverified, not assumed.

## Run the benchmark

The benchmark questions and verified evidence are in `benchmark/questions.json`.
Run dev first; this fixes the strategy selection using the predeclared rule
Evidence Hit@3, then Hit@1, then fewer chunks:

```powershell
python -m src.run --phase dev
python -m src.run --phase test
```

The test command is gated until a dev selection has been recorded. The default
run requires at least 30 useful content pages. For pipeline development only,
`python -m src.run --phase dev --allow-short-corpus` permits a preliminary run
and marks all its outputs as non-compliant. Do not report preliminary results as
meeting the corpus requirement.

The first run downloads
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions),
pinned to revision `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`.
All three strategies use the same local model and cosine distance:

| Strategy | Chunking | Retrieval |
|---|---|---|
| A | Fixed 300 characters, 40-character overlap | Dense top-3, no filter |
| B | Greedy source-line/paragraph grouping, maximum 300 characters | Dense top-3, no filter |
| C | Exactly the same chunks as B | Apply the question's `section` filter when present |

`results/` contains phase metrics, full top-3 traces, ingestion statistics and
audit sheets. The runner computes retrieval matches and generates blank audit
templates; the current coherence and grounding sheets contain clearly labeled
AI-assisted assessments, not independent human judgments. Retrieved metadata is
also checked against its source page.

## Data and submission notes

The originally supplied PDF has 20 physical pages; the pipeline excludes the
bibliography page when counting useful pages, leaving 19. The repository
GUIDE/RUBRIC specify 30 useful pages; the learner reports that the instructor
waived that threshold for this submission. The report records both the measured
count and that clarification; it does not claim the corpus has 30 pages.

The learner confirms permission to share the supplied PDF and its excerpts in
the public repository. The PDF itself does not identify a specific license or
original source URL; the manifest records the permission confirmation without
claiming an open license. Do not commit API keys, `.env`, private documents or
data you cannot redistribute.
