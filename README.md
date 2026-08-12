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
           -> bounded evidence windows -> hierarchical document digests
           -> bounded candidate retrieval -> cross-document evidence selection
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

Planning requests use finite, provider-neutral input budgets by default. Large
documents are digested through ordered evidence windows; retrieval shortlists
window-local candidates before global selection, so the selector never receives
the full corpus catalogue. Oversized text is split on semantic boundaries with
an exact character fallback, and oversized text-backed tables are split by rows;
both retain the original evidence ID and exact source values. Unsupported
oversized structured forms fail explicitly instead of being truncated.
`PlanningScaleConfig` accepts an explicit token counter, nested budgets, and an
optional retriever for offline evaluation or provider-specific deployment. A
single shared limiter bounds all planning calls.

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

### Semantic slide-content synthesis

After planning, deterministic slide contexts resolve each slide's selected
evidence into provider-safe semantic inputs. `generate_slide_contents()` then
creates validated `SlideContent` objects (text, lists, and source references
for charts, tables, images, and equations), without coordinates, styles, or
PPTX rendering. `PresentationContent` preserves exact outline order and can be
persisted with `write_presentation_content_json()` using the same explicit,
UTF-8, sorted-key, no-overwrite semantics as outline artifacts. The synthesis
live test is opt-in and requires both `OPENAI_API_KEY` and
`RUN_OPENAI_INTEGRATION_TESTS=1`.

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

### Generate a reviewable PowerPoint deck

The pipeline can now complete the full path from DOCX documents to an editable,
16:9 PowerPoint deck. The first renderer uses a fixed executive-policy visual
system and a finite layout vocabulary: title, section divider, headline/body,
key points, two-column, visual/text, table, and chart-focused slides. It writes
the PPTX alongside its validated outline, semantic content, physical layout,
source/generated assets, and a machine-readable rendering report.

```bash
OPENAI_API_KEY=... uv run presentation-pipeline generate docs/*.docx \
  --goal "Synthesize the health-policy material into an actionable briefing" \
  --audience "Senior health-policy leadership" \
  --slides 10 \
  --tone "concise, evidence-led, executive" \
  --output-dir output/pptx-evaluation \
  --generate-images \
  --image-concurrency 2
```

`--generate-images` is optional. When enabled, it can add at most three
OpenAI-generated decorative illustrations; they never replace source evidence,
tables, charts, numbers, logos, or source-document images. A failed image
request falls back to a text-first layout and is recorded in
`render-report.json`. Table-derived chart candidates become editable charts.
Native DOCX charts without extracted values become visible source summaries with
a warning rather than invented data. Candidate selection is deterministic before
any image request, and `--image-concurrency` only controls the bounded number
of simultaneous image API calls.

Deck outputs are written in a sibling staging directory and promoted only after
the PPTX passes an independent OOXML ZIP/XML/relationship/slide-count check
(with an additional LibreOffice conversion check when `soffice` is available).
An existing output directory is rejected unless `--overwrite` is set; an
overwrite keeps the prior directory intact until the new staged deck passes its
promotion gate. Audit JSON contains only published paths, never staging paths.

For the curated five-document healthcare workflow used by the offline
regression, run:

```bash
uv run --env-file .env presentation-pipeline generate \
  'docs/(醫管組)1150712_台灣醫事法律學會-AI智慧健保治理-談參資料(草案)_1150623奉核.docx' \
  'docs/1140910_偏鄉方案及保障(行政科).docx' \
  'docs/20260321_ 雲林國際居家醫療研討會-談參資料(規劃科).docx' \
  'docs/中央癌症防治會報第21次會議-備參資料(規劃科).docx' \
  'docs/南投縣衛生局智慧醫療雲備參(醫管組).docx' \
  --goal 'Synthesize digital-health governance, healthcare access, home care, cancer policy, and smart-care infrastructure into one actionable executive briefing.' \
  --audience 'Senior health-policy leadership' \
  --slides 10 \
  --tone 'concise, evidence-led, executive' \
  --presentation-type 'executive policy briefing' \
  --must-include 'cross-document priorities and actionable recommendations' \
  --output-dir 'output/five-docx-healthcare-briefing'
```

Append `--generate-images --max-generated-images 3 --image-concurrency 3`
to opt into decorative image generation. Add `--overwrite` only when the
entire existing output directory should be replaced.

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
