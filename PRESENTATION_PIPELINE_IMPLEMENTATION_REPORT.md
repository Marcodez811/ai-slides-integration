# Presentation Pipeline Implementation Report

Date: 2026-08-11

## 1. Executive summary

This implementation adds a new `presentation_pipeline` package beside the
existing `docx_pipeline`. The extraction package remains the source of truth
and was not modified.

The implemented stage accepts one or more DOCX paths and can produce a
provenance-validated `PresentationOutline` through this sequence:

```text
DOCX paths
    -> DocumentJob[]
    -> concurrent DOCX extraction
    -> DocumentArtifact[]
    -> deterministic DocumentIndex[]
    -> concurrent per-document DocumentDigest[]
    -> cross-document EvidenceSelection
    -> PresentationOutline
    -> deep provenance validation
```

The most important architectural boundary is now:

```text
ExtractionResult
    -> NormalizedBlock
    -> EvidenceItem
    -> DocumentDigest
    -> EvidenceSelection
    -> PresentationOutline
```

`SourceNode` remains the immutable audit/source layer. `NormalizedBlock`
remains the semantic view created by `docx_pipeline`. The new `EvidenceItem`
is the presentation reasoning layer and is the only layer that the planning
models reference directly.

The implementation is provider-neutral. It defines a `StructuredGenerator`
protocol but does not hardcode OpenAI, Gemini, Claude, or another provider.

## 2. Scope delivered

The implementation contains 22 Python files under `presentation_pipeline`:

```text
presentation_pipeline/
├── __init__.py
├── common/
│   ├── __init__.py
│   ├── ids.py
│   └── models.py
├── corpus/
│   ├── __init__.py
│   ├── batch.py
│   ├── manifest.py
│   └── models.py
├── indexing/
│   ├── __init__.py
│   ├── builder.py
│   └── models.py
├── understanding/
│   ├── __init__.py
│   ├── models.py
│   ├── prompts.py
│   └── service.py
├── planning/
│   ├── __init__.py
│   ├── models.py
│   ├── prompts.py
│   └── service.py
├── validation/
│   ├── __init__.py
│   └── provenance.py
└── pipeline.py
```

Other repository changes:

- `pyproject.toml` now packages both `docx_pipeline` and
  `presentation_pipeline` in the wheel.
- `README.md` documents the new planning stage and its provider-neutral API.
- Three test modules cover batching, indexing, understanding, planning,
  validation, concurrency, and orchestration.

## 3. Shared contracts and identity rules

### 3.1 Strict Pydantic base model

All presentation-layer models inherit from `PipelineModel`, which configures:

```python
ConfigDict(extra="forbid")
```

Unknown fields therefore fail validation instead of being silently ignored.
This is intended to prevent schema drift between deterministic code, LLM
structured outputs, stored manifests, and downstream slide generation.

### 3.2 Two kinds of identifiers

The implementation intentionally separates execution IDs from semantic IDs.

Execution and persistence records use UUID-based helpers:

- `make_job_id()`
- `make_batch_id()`
- `make_corpus_id()`

Semantic presentation objects use:

```python
make_pipeline_id(kind, *components)
```

This function length-prefixes each component, hashes the encoded data with
SHA-256, and returns a stable 24-hex-character suffix. Length-prefixing avoids
ambiguous inputs such as `("ab", "c")` and `("a", "bc")`.

Evidence IDs include the document ID, indexer version, evidence kind, block
ID, derivation type, and source node IDs. Rebuilding the same index from the
same extraction result therefore produces the same evidence IDs.

## 4. Stage 1: corpus and batch extraction

### 4.1 `DocumentJob`

Each input document becomes a strict `DocumentJob` containing:

- `job_id`
- `input_path`
- `extraction_config`
- optional `asset_output_dir`

The canonical document ID is deliberately absent at this point. It is only
known after `extract_docx()` reads the actual DOCX bytes.

### 4.2 Job construction

`build_jobs()` preserves input order and creates one job per path.

When an asset output root is provided, every job receives a separate child
directory named after its job ID:

```text
asset root/
├── job-.../
├── job-.../
└── job-.../
```

This avoids multiple extraction workers writing into the same asset
directory.

### 4.3 `DocumentArtifact`

A successful extraction becomes a `DocumentArtifact` containing:

- `job_id`
- canonical `doc_id`
- `filename`
- `health`
- the complete in-memory `ExtractionResult`

The model validates that `doc_id` and `filename` match the embedded
`ExtractionResult.document` metadata.

Artifact health is:

- `READY` when no silent loss is reported;
- `DEGRADED` when silent loss exists and degraded output was explicitly
  allowed.

A ready artifact cannot contain silent losses, and a degraded artifact must
contain at least one silent loss.

### 4.4 Failure representation

`DocumentFailure` captures a failed job without losing the rest of the batch:

- `job_id`
- `input_path`
- failing `stage`
- `error_type`
- human-readable `message`
- any associated extraction diagnostics

Extractor exceptions are converted into structured failures with a synthetic
error diagnostic. Error diagnostics returned by the extractor also produce a
failure.

### 4.5 Threaded execution

`extract_batch()` uses:

```python
ThreadPoolExecutor(max_workers=min(max_workers, len(jobs)))
```

Important behavior:

- `max_workers` must be a real integer greater than or equal to one; booleans,
  strings, floats, zero, and negative values are rejected.
- duplicate `job_id` values are rejected before work begins;
- an empty batch returns a valid empty `BatchExtractionResult`;
- all extraction jobs run independently;
- an exception in one job does not cancel other jobs;
- externally visible documents and failures retain input-relative order rather
  than completion order.

The batch result contains:

```python
BatchExtractionResult(
    batch_id=...,
    documents=[...],
    failures=[...],
)
```

`artifacts` is exposed only as a read-only compatibility alias for
`documents`; `documents` is the canonical serialized field.

### 4.6 Strict and degraded modes

Default batch behavior is strict:

- any error diagnostic fails the document;
- any silent loss fails the document.

With `allow_degraded=True`, silent loss produces a degraded artifact instead
of a failure. Error diagnostics still fail even in degraded mode.

The top-level `generate_outline()` currently uses strict mode.

## 5. Corpus persistence boundary

Runtime processing keeps complete `ExtractionResult` objects in memory.
Persistence is represented separately by:

```python
DocumentArtifactRef(
    doc_id=...,
    filename=...,
    artifact_uri=...,
    health=...,  # optional
)
```

`CorpusManifest` contains a `corpus_id`, schema version, and ordered document
references. Duplicate document IDs are rejected.

This is a contract only. The implementation does not choose local disk, S3,
a database, or another persistence backend, and it does not yet implement a
manifest writer/loader service.

## 6. Stage 2A: deterministic document indexing

The index is the main new deterministic abstraction. No LLM participates in
this stage.

### 6.1 Evidence kinds

The following evidence kinds are supported:

- `TITLE`
- `TEXT`
- `CAPTION`
- `LIST`
- `TABLE`
- `IMAGE`
- `EQUATION`
- `CHART`
- `CHART_CANDIDATE`

Unsupported/header/footer/index-only normalized blocks are not promoted to
presentation evidence.

### 6.2 `EvidenceItem`

Every evidence item contains:

- `doc_id`
- deterministic `evidence_id`
- `kind`
- one or more normalized `block_ids`
- one or more `source_node_ids`
- zero or more `section_ids`
- zero or more `asset_ids`
- optional human-readable `text`
- JSON-safe `structured_data`

This creates the provenance chain:

```text
evidence_id
    -> block_ids
    -> source_node_ids
    -> SourceNode.source
    -> OOXML part and XML path
```

### 6.3 `IndexedSection`

Normalized sections are copied into deterministic section records containing:

- section identity and hierarchy;
- title and level;
- original block membership;
- evidence IDs created from those blocks;
- section detector confidence and metadata.

Nested sections can intentionally overlap because the existing section
normalizer lets parent sections retain nested-section content.

### 6.4 `DocumentIndex`

Each index records:

- presentation index schema version;
- `doc_id` and filename;
- extraction schema and extractor versions;
- indexer version;
- ordered sections;
- ordered evidence.

The index is a semantic catalogue, not a summary. It preserves enough
structure for later retrieval and slide planning without requiring an LLM to
handle raw source nodes.

### 6.5 Pre-index validation

`build_document_index()` fails before producing evidence when it finds broken
relationships, including:

- artifact document ID or filename disagreeing with the extraction result;
- duplicate or missing node, block, asset, or section IDs;
- a normalized block referencing unknown source nodes;
- a table block whose first source is not a table node;
- an image block whose first source is not an image node;
- a non-null image asset ID that does not exist in `ExtractionResult.assets`;
- disagreement between an image block asset ID and its image source node;
- a section referencing an unknown heading, parent section, or block;
- multiple table blocks mapped to the same table node;
- a chart candidate without a corresponding normalized table block.

An unresolved image with `asset_id=None` is retained as image evidence with an
empty `asset_ids` list. This is useful because the extraction diagnostics still
explain why the asset could not be resolved.

### 6.6 Text, title, caption, table, equation, and chart evidence

The builder uses the actual normalized payload contracts:

- title/text/caption: `payload["content"]`;
- table: `payload["text"]`;
- equation: `payload["latex"]`;
- chart: the native chart payload is retained as structured data;
- image: the normalized image payload plus validated asset metadata is
  retained as structured data.

### 6.7 List reconstruction

The normalized list payload contains list structure and source paragraph IDs,
but does not copy item text. The implementation therefore reconstructs list
text deterministically:

1. validate every nested list item structure;
2. require every item source ID to belong to the normalized block;
3. require every item source to be a paragraph;
4. traverse paragraph descendants in source sequence order;
5. collect non-deleted text runs, line breaks, and tabs;
6. join paragraph text in list source order.

The nested list structure is retained in `structured_data`, while the rebuilt
plain text is available in `EvidenceItem.text`.

### 6.8 Table-derived chart candidates

Every structural table is passed through the existing side-effect-free
`profile_tables()` enrichment.

For an eligible table, indexing emits two consecutive evidence items:

```text
TABLE evidence
CHART_CANDIDATE evidence
```

The derived chart candidate includes:

- category column;
- categories;
- named series and numeric values;
- original source cell node IDs in the internal index;
- the normalized table block ID.

This means the planner can distinguish “the source contains a table” from
“that table can safely become a chart.”

### 6.9 Determinism and immutability

Evidence follows normalized block order. A derived chart candidate is inserted
immediately after its table. IDs are deterministic, and indexing never mutates
the extraction result.

All structured values are converted into JSON-safe forms. Non-finite floats or
unsupported value types fail fast instead of leaking into an LLM request or
serialized index.

## 7. Stage 2B: document understanding

### 7.1 Provider-neutral generator protocol

The pipeline depends on this interface:

```python
class StructuredGenerator(Protocol):
    async def generate(
        self,
        *,
        system_prompt: str,
        input_data: dict[str, object],
        response_model: type[TStructured],
    ) -> TStructured | dict[str, object]:
        ...
```

An adapter is responsible for asking its provider for structured output that
matches `response_model`.

The core package does not import a provider SDK.

### 7.2 Digest models

The understanding layer defines:

- `EvidenceRef`: document ID plus one or more unique evidence IDs;
- `TopicDigest`: topic, summary, and supporting evidence;
- `KeyFact`: claim and supporting evidence;
- `DocumentDigest`: document summary, topics, and key facts.

Blank identifiers and duplicate evidence IDs within a reference are rejected
by schema validation.

### 7.3 LLM input boundary

The digest generator does not receive the `ExtractionResult`, raw
`SourceNode`s, XML paths, relationships, or normalized block IDs.

It receives:

```text
document metadata
compact section hierarchy
evidence_id + kind + section context
human-readable text
sanitized structured content
```

Structured chart and table information—such as categories, series, and
values—is retained. Mechanical provenance fields are recursively removed,
including source node IDs, table node IDs, block IDs, XML/relationship/path
fields, asset IDs, and internal chart candidate IDs.

This keeps enough semantic data for reasoning while keeping provenance IDs
under deterministic program control.

### 7.4 Digest concurrency

`generate_digests()`:

- indexes `DocumentIndex` objects by document ID;
- rejects missing or duplicate document indexes;
- validates concurrency as an integer greater than or equal to one;
- uses `asyncio.Semaphore(concurrency)` to bound remote calls;
- uses `asyncio.gather()` to preserve artifact input order.

Each returned digest must use the same `doc_id` as its artifact.

## 8. Stage 3A: cross-document evidence selection

### 8.1 Requirements

`PresentationRequirements` contains:

- `goal`
- `audience`
- `target_slide_count` from 1 to 100
- optional `tone`
- optional `presentation_type`
- `must_include`
- `must_avoid`

Blank requirement strings are rejected.

### 8.2 Selection output

Each `SelectedEvidence` contains:

- `doc_id`
- one `evidence_id`
- selection `reason`

`EvidenceSelection` contains a non-empty `selected` list and an optional
high-level strategy. Duplicate `(doc_id, evidence_id)` selections are rejected.

### 8.3 Selection input

The selector receives:

- presentation requirements;
- all document digests;
- a compact per-document evidence catalogue;
- compact section hierarchies;
- sanitized structured evidence content.

It does not receive raw source or OOXML internals.

The prompt instructs the model not to fabricate facts, alter numbers, or
invent/mutate evidence IDs.

## 9. Stage 3B: presentation outline

### 9.1 Outline models

Supported slide purposes are:

- `TITLE`
- `SECTION`
- `CONTENT`
- `SUMMARY`

Supported semantic content forms are:

- `TEXT`
- `LIST`
- `TABLE`
- `CHART`
- `IMAGE`
- `MIXED`

A `SlideOutline` contains:

- `slide_id`
- title;
- purpose;
- one main message;
- content requirements;
- evidence references;
- preferred semantic content forms.

It deliberately contains no coordinates, font sizes, dimensions, or renderer
instructions.

An `OutlineSection` contains its own ID, title, purpose, and ordered slides.
`PresentationOutline` contains schema version, title, objective, narrative,
and ordered sections. Slide IDs must be unique across the complete outline.

### 9.2 Outline input

The outline generator receives:

- requirements;
- document digests;
- only the evidence selected in Stage 3A;
- each selected item's semantic text/structured content;
- the evidence selection reason.

It does not receive the complete corpus again.

### 9.3 Validation and repair retry

The initial outline is schema-validated and then provenance-validated.

If provenance validation fails, the implementation performs one repair retry.
The retry prompt contains the validation error and tells the generator to
regenerate using valid references. The invalid output is not silently edited
or stripped.

If the second result is invalid, the validation error is raised.

The repair retry currently applies to provenance validation failures. Provider
errors and ordinary Pydantic schema errors are surfaced directly.

## 10. Deep provenance validation

### 10.1 Corpus lookup construction

`CorpusLookup.from_artifacts_indexes()` requires a one-to-one mapping between
document artifacts and document indexes.

It rejects:

- duplicate artifacts;
- duplicate indexes;
- an artifact without an index;
- an index without an artifact;
- artifact/extraction document ID disagreement.

For each document it records the valid sets of:

- normalized block IDs;
- source node IDs;
- asset IDs;
- section IDs;
- evidence IDs.

Every evidence item is deep-validated while the lookup is built.

### 10.2 Evidence validation

The validator proves that:

- the evidence item belongs to the indexed document;
- every block ID exists in the extraction result;
- every source node ID exists in the extraction result;
- every asset ID exists in the extraction result;
- every section ID exists in the extraction result;
- every generated evidence reference names a real document and evidence item.

### 10.3 Digest validation

Digest validation rejects:

- duplicate digests for the same document;
- unknown document IDs;
- unknown evidence IDs;
- a per-document digest citing evidence from another document.

### 10.4 Selection validation

Every selected `(doc_id, evidence_id)` pair must resolve through the corpus
lookup.

### 10.5 Outline validation

All slide evidence references must resolve. In addition:

- every `CONTENT` slide must contain evidence;
- every `SUMMARY` slide must contain evidence;
- `TITLE` and `SECTION` slides may omit evidence.

### 10.6 What provenance validation does not prove

The validator proves referential integrity. It does not yet prove semantic
entailment.

For example, it can prove that a slide cites a real revenue table, but it does
not independently prove that every word of the slide message is entailed by
that table. It also does not independently recalculate derived claims such as
“revenue increased 18%.” Those are future claim-verification concerns.

## 11. Top-level orchestration

The public entry point is:

```python
await generate_outline(
    input_paths,
    requirements,
    generator,
    extraction_workers=4,
    llm_concurrency=4,
)
```

The exact flow is:

1. reject an empty input path list;
2. build ordered document jobs;
3. extract documents through the thread pool;
4. raise `BatchExtractionError` if any document failed;
5. build one deterministic index per artifact;
6. build the corpus provenance lookup;
7. generate document digests with bounded async concurrency;
8. validate all digests;
9. select cross-document evidence;
10. validate the selection;
11. generate the outline;
12. validate it and retry once on invalid provenance;
13. return the valid `PresentationOutline`.

For `N` documents, the normal LLM-call pattern is:

```text
N digest calls, bounded and concurrent
1 evidence-selection call
1 outline call
0 or 1 outline-repair call
```

Batch extraction isolates individual failures so all jobs can finish, but the
top-level outline workflow is all-or-nothing: any batch failure raises
`BatchExtractionError` before indexing or LLM work begins.

## 12. Concurrency boundaries

The implementation uses different concurrency models for different work:

```text
DOCX extraction       -> ThreadPoolExecutor
Document indexing     -> synchronous deterministic loop
Document digests      -> asyncio + Semaphore
Evidence selection    -> one awaited structured call
Outline generation    -> one awaited structured call
```

This matches the workload types: local document extraction is independent
blocking work, while model calls are remote asynchronous I/O.

## 13. Error behavior summary

| Condition | Result |
|---|---|
| Invalid worker count | `ValueError` before extraction |
| Duplicate job ID | `ValueError` before extraction |
| Individual extractor exception | `DocumentFailure`; other jobs continue |
| Extractor error diagnostic | `DocumentFailure` |
| Silent loss in strict mode | `DocumentFailure` |
| Silent loss with degraded mode | `DEGRADED` artifact |
| Any failure in top-level workflow | `BatchExtractionError` before LLM work |
| Invalid index relationship | `ValueError` during deterministic indexing |
| Missing artifact/index pairing | `ProvenanceValidationError` |
| Invalid digest/selection reference | `ProvenanceValidationError` |
| Invalid outline reference | one repair retry, then error |
| Generator/provider error | propagated to caller |
| Structured response schema error | Pydantic validation error propagated |

## 14. Test coverage

The repository test suite currently passes:

```text
61 passed
```

The presentation implementation adds 26 collected test cases across three
files.

### 14.1 Batch tests

Coverage includes:

- input-order preservation despite out-of-order worker completion;
- exception isolation;
- strict silent-loss rejection;
- degraded-mode acceptance;
- error diagnostics remaining fatal in degraded mode;
- empty batch behavior;
- worker-count validation;
- strict extra-field rejection;
- manifest validation;
- deterministic, ambiguity-safe pipeline IDs.

### 14.2 Indexing tests

Coverage includes:

- deterministic repeated output;
- no extraction-result mutation;
- title, text, list, table, image, equation, and native chart evidence;
- table-derived chart candidates;
- list text reconstruction;
- section/evidence membership;
- block, node, asset, and evidence resolution;
- unresolved image behavior;
- dangling image asset rejection;
- malformed table-source mapping rejection.

### 14.3 Understanding, planning, and validation tests

Coverage includes:

- compact prompt input without source/XML internals;
- structured chart values preserved for model reasoning;
- digest concurrency limits and deterministic ordering;
- invalid concurrency rejection;
- unknown document/evidence rejection;
- dangling block, source node, and asset rejection;
- cross-document digest evidence rejection;
- content and summary slide evidence requirements;
- end-to-end orchestration with a fake structured generator;
- empty top-level input rejection.

The original 35 extraction tests continue to pass, confirming that the new
sibling package did not regress `docx_pipeline`.

## 15. Key deviations and refinements from the initial proposal

The proposal was implemented with these repository-specific refinements:

1. `presentation_pipeline` was added to Hatch's wheel package list; otherwise
   the new package would work locally but be omitted from built distributions.
2. List text is reconstructed from referenced paragraph descendants because
   the normalized list payload stores structure and source IDs, not copied
   text.
3. Direct native charts and equations are indexed, in addition to
   table-derived chart candidates.
4. Asset output is separated by job when a shared asset root is provided.
5. Batch and LLM concurrency values are explicitly validated.
6. Structured chart/table values are preserved at the LLM boundary while
   mechanical provenance fields are stripped.
7. Provenance checks were extended beyond blocks and nodes to document
   ownership, sections, and assets.
8. The outline stage uses one explicit regeneration retry rather than silently
   deleting invalid evidence references.

## 16. Deliberately out of scope

The following are not implemented in this stage:

- a concrete OpenAI, Gemini, Claude, or other provider adapter;
- API keys, provider configuration, rate-limit backoff, or provider telemetry;
- automatic prompt chunking for exceptionally large document indexes;
- persistence implementation for artifacts/manifests;
- a presentation-planning CLI;
- image semantic classification integration;
- claim-level semantic entailment or independent arithmetic verification;
- Slide JSON synthesis;
- layout/composition decisions;
- PPTX rendering;
- coordinates, dimensions, fonts, themes, or design templates.

These omissions preserve the intended boundary. The current output is a
validated semantic presentation outline, not a rendered deck.

## 17. Practical usage

```python
from presentation_pipeline import generate_outline
from presentation_pipeline.planning import PresentationRequirements

requirements = PresentationRequirements(
    goal="Explain Q2 performance",
    audience="Executive leadership",
    target_slide_count=12,
    tone="concise",
    presentation_type="business review",
    must_include=["revenue performance"],
)

outline = await generate_outline(
    ["financials.docx", "market-analysis.docx"],
    requirements,
    generator=my_structured_generator,
    extraction_workers=4,
    llm_concurrency=4,
)
```

The injected generator must implement the keyword-only `StructuredGenerator`
contract. The returned outline is guaranteed to pass schema and referential
provenance validation. It is not guaranteed to have passed semantic
entailment verification because that stage does not yet exist.

## 18. Recommended next implementation stage

The next clean boundary is:

```text
PresentationOutline
+ selected and resolved EvidenceItem objects
+ original normalized blocks/assets
    -> Slide synthesis
    -> Slide JSON[]
    -> layout/composition
    -> renderer
```

Before implementing Slide JSON, the most valuable additions would be:

1. one concrete `StructuredGenerator` adapter with retry/backoff and usage
   telemetry;
2. prompt-size estimation and deterministic evidence chunking;
3. claim-level numeric verification for important slide messages;
4. persisted stage artifacts for reproducibility and resume support;
5. optional image classification enrichment before evidence selection.

## 19. Final status

The implemented stage is complete through validated `PresentationOutline`.
It is deterministic up to the provider-dependent reasoning stages, preserves
the existing extraction source of truth, isolates the LLM from mechanical
provenance internals, and maintains a resolvable audit trail back to normalized
blocks, source nodes, sections, assets, and OOXML source locations.

Verification at report generation time:

```text
61 tests passed
presentation_pipeline compiled successfully
git diff --check passed
```
