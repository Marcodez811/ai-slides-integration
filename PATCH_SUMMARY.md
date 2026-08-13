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

## Digest scope and bounded hydration follow-up

- `DocumentDigest.scoped_to(...)` now rebuilds a non-mutating digest view whose
  references contain only evidence identities available to the current stage.
- Evidence-selection prompts scope document context to candidate identities;
  outline prompts scope it again to selected identities. Documents not
  represented at a stage are omitted, while strict provenance validation is
  unchanged.
- Evidence-selection hydration now returns only candidates that retain bounded
  source transport, so prompt construction cannot reload canonical evidence for
  an empty candidate transport.
- Hydration and global-cap telemetry report candidate counts before and after
  bounded transport retention.
- Validation for this follow-up: focused retrieval, reduction, requirements,
  observability, and logging tests passed (`61 passed`); the maintained suite
  passed (`247 passed, 4 skipped`) when excluding the stale module below.
- The maintained suite still excludes `tests/test_understanding_contracts.py`:
  it fails collection because it imports the obsolete
  `presentation_pipeline.understanding.contracts.digest_output_contract` API,
  which no longer exists after deterministic digest-contract migration. This is
  a stale test-module compatibility blocker, not a passing-suite claim.
