# Benchmark protocol

`questions.json` contains 20 questions with verified source-page evidence, split
before retrieval into 15 dev and 5 test questions. Four questions provide a
`section` filter, including one in test. The filter values come from the PDF's
chapter headings; they are not derived from gold pages at query time.

Run `python -m src.run --phase dev` before `--phase test`. The dev phase records a
selection using the fixed rule in the experiment log. The test phase refuses to
run without that dev selection. If corpus content or strategy settings change
after selection, rerun dev and record the change before evaluating test.

Automatic evidence matching requires the gold source/page and the exact verified
quote to appear in the retrieved chunk. Human review may identify equivalent
evidence, but must be recorded separately and must not silently overwrite the
automatic trace. Coherence, metadata correctness and grounding scores are not
inferred by code.

The coherence sheet samples ten evenly spaced content pages and points to the
first non-header source line of at least 60 characters; the same source position
is used for A, B and C. Fill in the requested score and rationale after reading
the chunk in context.
