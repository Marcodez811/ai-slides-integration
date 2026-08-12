"""One-shot bounded digest generation with exactly one safe repair."""
from __future__ import annotations

from typing import TypeVar

from presentation_pipeline.budgeting import GenerationLimiter, InputBudget
from presentation_pipeline.observability import safe_event

from .contracts import DigestNode, DigestOutputContract, DigestOutputContractError, estimate_digest_transport_tokens, validate_digest_contract
from .prompts import DIGEST_CONTRACT_REPAIR_PROMPT
from .reduction import DigestProvenanceError, validate_digest_scope

T = TypeVar("T", bound=DigestNode)


async def generate_bounded_digest(*, limiter: GenerationLimiter, system_prompt: str,
                                  input_data: dict[str, object], response_model: type[T], stage: str,
                                  input_budget: InputBudget, output_contract: DigestOutputContract,
                                  allowed_evidence_ids: set[str], expected_doc_id: str,
                                  expected_window_id: str | None = None,
                                  include_contract_in_payload: bool = True) -> T:
    """Call once, validate fully, then repair exactly once without source input."""
    payload = {**input_data, "output_contract": output_contract.provider_value()} if include_contract_in_payload else dict(input_data)
    prompt = system_prompt if include_contract_in_payload else system_prompt + f"\nLIMIT:{output_contract.max_transport_tokens}/{output_contract.max_topics}/{output_contract.max_key_facts}"
    initial = await limiter.invoke(system_prompt=prompt, input_data=payload, response_model=response_model,
                                   budget=input_budget, stage=stage)
    if not isinstance(initial, response_model):
        raise TypeError(f"{stage} did not return {response_model.__name__}")
    try:
        return _validate(initial, response_model, stage, limiter, output_contract, allowed_evidence_ids,
                         expected_doc_id, expected_window_id, False)
    except DigestOutputContractError as violation:
        safe_event("digest_contract_violation", **violation.metadata, window_id=expected_window_id)
        safe_event("digest_contract_repair_start", **violation.metadata, window_id=expected_window_id)
        repair_payload = {"digest": initial.model_dump(mode="json"), "output_contract": output_contract.provider_value()}
        repaired = await limiter.invoke(system_prompt=DIGEST_CONTRACT_REPAIR_PROMPT, input_data=repair_payload,
                                        response_model=response_model, budget=input_budget, stage="digest_contract_repair")
        if not isinstance(repaired, response_model):
            raise TypeError(f"{stage} repair did not return {response_model.__name__}")
        try:
            value = _validate(repaired, response_model, stage, limiter, output_contract, allowed_evidence_ids,
                              expected_doc_id, expected_window_id, True)
        except DigestOutputContractError as failure:
            failure.metadata["original_reason"] = violation.metadata["reason"]
            failure.metadata["reason"] = "repair_failed_contract"
            safe_event("digest_contract_repair_failed", **failure.metadata, window_id=expected_window_id)
            raise
        safe_event("digest_contract_repair_success", doc_id=expected_doc_id, stage=stage,
                   response_model=response_model.__name__, window_id=expected_window_id,
                   estimated_transport_tokens=estimate_digest_transport_tokens(value, limiter.token_counter),
                   target_tokens=output_contract.max_transport_tokens, repaired=True)
        return value


def _validate(result: T, model: type[T], stage: str, limiter: GenerationLimiter,
              contract: DigestOutputContract, allowed: set[str], doc_id: str,
              window_id: str | None, repaired: bool) -> T:
    tokens = estimate_digest_transport_tokens(result, limiter.token_counter)
    def failure(reason: str) -> DigestOutputContractError:
        return DigestOutputContractError(doc_id=doc_id, stage=stage, response_model=model.__name__,
            estimated_transport_tokens=tokens, contract=contract, topic_count=len(result.topics),
            key_fact_count=len(result.key_facts), repair_attempted=repaired, reason=reason)
    if result.doc_id != doc_id:
        raise failure("document_id_mismatch")
    if window_id is not None and getattr(result, "window_id", None) != window_id:
        raise failure("window_id_mismatch")
    try:
        validate_digest_scope(result, doc_id=doc_id, allowed_evidence_ids=allowed)
    except DigestProvenanceError:
        raise failure("provenance_outside_scope") from None
    validate_digest_contract(result, contract=contract, token_counter=limiter.token_counter,
                             stage=stage, repair_attempted=repaired)
    safe_event("digest_contract_valid", doc_id=doc_id, stage=stage, response_model=model.__name__,
               window_id=window_id, estimated_output_tokens=tokens, target_tokens=contract.max_transport_tokens,
               topic_count=len(result.topics), key_fact_count=len(result.key_facts), repaired=repaired)
    return result
