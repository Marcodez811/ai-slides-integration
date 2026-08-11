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

The LLM boundary is provider-neutral. Supply an implementation of
`StructuredGenerator`; no OpenAI, Gemini, or Claude adapter is hardcoded:

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
