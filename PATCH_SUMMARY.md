# Patch summary

This copy was modified to remove provider-based iterative digest compaction from the active document-understanding pipeline.

Key changes:

- Added deterministic `DigestOutputContract` enforcement.
- Verbose model digests are bounded locally after one generation call; no LLM shrink/retry loop is used.
- Digest nodes are capped to 25% of the configured reduction input budget so adjacent nodes remain pairable with generous headroom.
- Simplified hierarchical reduction: bounded grouping, singleton carry-forward, bounded merges, final reduction.
- Added `ReductionInvariantError` for application sizing bugs instead of attempting repeated model compaction.
- Preserved strict evidence provenance validation.
- Kept legacy `ReductionNonReductionError` constructor compatibility for existing failure-report tooling.
- Replaced obsolete compaction tests with output-contract and pairability tests.

Validation in the sandbox:

- `210 passed, 6 skipped`
- `python -m compileall` passed for both `presentation_pipeline` and `docx_pipeline`.
- `test_real_docx_reaches_a_reopenable_pptx_with_audit_artifacts` passed.

Note: the uploaded ZIP did not contain the newer `understanding/contracts.py` / `understanding/generation.py` shown in the user's latest local traceback, so these files were implemented directly into the uploaded baseline.

## Retrieval scope fix

- Local candidate discovery now receives document-level semantic context with provenance IDs stripped.
- Out-of-scope/duplicate/over-limit local candidates are filtered and logged instead of cancelling the whole concurrent retrieval batch.
- Multiple slices with the same canonical evidence ID inside one local window are preserved together.
- Strict identity validation remains in later retrieval/planning provenance boundaries.
- Full suite after this patch: 212 passed, 6 skipped.
