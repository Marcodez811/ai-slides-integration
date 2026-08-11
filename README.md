# Auditable DOCX extraction for slide generation

This repository contains a provenance-preserving DOCX pipeline. The new
`docx_pipeline` package reads OOXML in source order and emits:

- a flat, deterministic Source IR (`nodes`);
- content-addressed image assets;
- an independent feature inventory and coverage report;
- structured diagnostics instead of silent skips;
- disposable semantic block, list, section, and caption views.

`docx_pipeline` is the canonical extraction interface and result contract.

## Presentation planning pipeline

`presentation_pipeline` builds on that immutable extraction result without
changing `docx_pipeline`:

```text
DOCX files -> batch extraction -> deterministic document indexes
           -> document digests -> cross-document evidence selection
           -> presentation outline -> provenance validation
```

The deterministic index maps presentation-facing evidence IDs back to
normalized block IDs and source node IDs. Tables can produce both table
evidence and chart-candidate evidence. List text is reconstructed from its
referenced source paragraphs, and images retain validated asset references.

The LLM boundary is provider-neutral. The orchestration code depends only on
the `StructuredGenerator` contract, so a provider adapter can be substituted
without changing extraction, indexing, planning, or provenance validation:

```python
from presentation_pipeline import generate_outline
from presentation_pipeline.planning import PresentationRequirements

outline = await generate_outline(
    ["financials.docx", "market.docx"],
    PresentationRequirements(
        goal="Explain quarterly performance",
        audience="Executives",
        target_slide_count=12,
    ),
    generator,
)
```

Generated digest, selection, and outline references are rejected if they do
not resolve through the document index to real normalized blocks, source
nodes, and assets. Content and summary slides must cite evidence.

### Generate an outline with OpenAI

The included OpenAI adapter is one implementation of that boundary. Install
the project (including its OpenAI dependency), provide an API key, then run the
example with one or more DOCX files:

```bash
uv sync --group dev

OPENAI_API_KEY=... uv run python examples/generate_outline_openai.py \
  briefing.docx \
  --goal "Brief leadership on quarterly results" \
  --audience "Leadership team" \
  --slides 3 \
  --tone executive \
  --output output/outline.json
```

`OPENAI_MODEL` is optional and defaults to `gpt-5.6-terra`; set it to choose a
different compatible OpenAI model. The command never prints document text. It
writes a UTF-8, pretty, deterministic JSON representation of the validated
`PresentationOutline`. Output paths are explicit, their parent directories are
created as needed, and an existing file is protected unless `--overwrite` is
provided.

The generated JSON contains the deck title, objective, narrative, ordered
sections and slides, recommended content forms, and evidence IDs that remain
traceable to the source corpus. It is the hand-off artifact for the future
slide-rendering stage; it is not a PPTX file.

The real provider smoke test is intentionally opt-in and is skipped in normal
test runs (so it makes no network calls or model spend):

```bash
OPENAI_API_KEY=... RUN_OPENAI_INTEGRATION_TESTS=1 \
  uv run pytest -q tests/integration/test_openai_outline_live.py
```

For the locally supplied, gitignored `docs/*.docx` corpus, first run the
deterministic extraction/index/provenance audit. This makes no OpenAI calls and
writes `output/openai-docs-corpus/deterministic-report.json`:

```bash
RUN_DOCS_CORPUS_TESTS=1 \
  uv run pytest -q -s \
  tests/integration/test_openai_docs_corpus_live.py::test_docs_corpus_extracts_indexes_and_preserves_provenance
```

Then smoke-test the smallest real document with credentials loaded from the
local `.env` file:

```bash
RUN_OPENAI_INTEGRATION_TESTS=1 OPENAI_DOCS_LIMIT=1 \
  uv run --env-file .env pytest -q -s \
  tests/integration/test_openai_docs_corpus_live.py::test_openai_generates_validated_outline_from_docs_corpus
```

Remove `OPENAI_DOCS_LIMIT=1` to evaluate all documents together as one corpus.
The full run writes `outline.json` and `openai-report.json` beside the
deterministic report. Optional overrides include `OPENAI_MODEL`,
`OPENAI_DOCS_TARGET_SLIDES`, `OPENAI_DOCS_LLM_CONCURRENCY`, and
`OPENAI_DOCS_OUTPUT_DIR`. Normal `uv run pytest` runs still skip both corpus
tests and never read `.env` automatically.

## Python API

```python
from docx_pipeline.api import extract_docx

result = extract_docx(
    "input.docx",
    asset_output_dir="output/assets",
)

print(result.coverage.silent_losses)
print(result.canonical_json())
```

Deterministic table/chart profiling is a separate, side-effect-free enrichment:

```python
from docx_pipeline.enrichment import profile_tables

profiles = profile_tables(result.nodes)
chart_candidates = [profile.chart_candidate for profile in profiles if profile.chart_candidate]
```

Image classification is exposed as an `ImageClassifier` protocol and a
versioned cache contract. A concrete remote or local classifier is intentionally
injected after extraction; the mechanical parser makes no network calls.

The original DOCX is never modified. Omitting `asset_output_dir` performs no
asset filesystem writes; asset metadata and occurrence nodes are still
returned.

## CLI

```bash
uv run docx-pipeline extract input.docx \
  --output output/document.json \
  --assets-dir output/assets

uv run docx-pipeline validate output/document.json
```

For troubleshooting, both commands accept `--log-level` (for example
`DEBUG`), `--log-file output/docx-pipeline.log`, and `--log-json`. The CLI
logs operational counts and timings only; it does not log extracted document
text or asset contents.

Both commands return a non-zero status when the saved coverage report contains
silent losses.

## Tests

```bash
uv run pytest -q
```

The suite includes generated OOXML fixtures plus dynamic checks against the
locally supplied corpus. It covers mixed text/images, duplicate image
occurrences, equations, numbering facts, structural one-column tables, SDTs,
hyperlinks, text boxes, VML/Markup Compatibility fallbacks, package safety,
charts, deterministic output, normalization, CLI round-trips, and Source IR
invariants.
