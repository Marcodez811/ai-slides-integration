"""Hard, content-safe output bounds for every generated digest node."""

from __future__ import annotations

from dataclasses import dataclass

from presentation_pipeline.budgeting import InputBudget, TokenCounter

from .models import ChunkDigest, DigestFragment, DocumentDigest


DigestNode = ChunkDigest | DigestFragment | DocumentDigest


@dataclass(frozen=True, slots=True)
class DigestOutputContract:
    max_transport_tokens: int
    max_topics: int = 8
    max_key_facts: int = 12

    def provider_value(self) -> dict[str, object]:
        return {
            "max_transport_tokens_estimate": self.max_transport_tokens,
            "max_topics": self.max_topics,
            "max_key_facts": self.max_key_facts,
            "priority": "retain only presentation-relevant information",
        }


class DigestOutputContractError(ValueError):
    """A generated digest violated its bounded-output contract."""

    def __init__(self, *, doc_id: str, stage: str, response_model: str,
                 estimated_transport_tokens: int, contract: DigestOutputContract,
                 topic_count: int, key_fact_count: int, repair_attempted: bool,
                 reason: str) -> None:
        self.metadata = {
            "doc_id": doc_id, "stage": stage, "response_model": response_model,
            "estimated_transport_tokens": estimated_transport_tokens,
            "max_transport_tokens": contract.max_transport_tokens,
            "topic_count": topic_count, "max_topics": contract.max_topics,
            "key_fact_count": key_fact_count, "max_key_facts": contract.max_key_facts,
            "repair_attempted": repair_attempted, "reason": reason,
        }
        super().__init__(f"digest output contract violated ({reason})")


def digest_output_contract(budget: InputBudget) -> DigestOutputContract:
    """Use generous reduction-payload headroom for two maximum digest nodes."""
    limit = budget.target_input_tokens_or_usable
    # The reduction wrapper, JSON punctuation, and prompt all consume space.
    # A fifth of input capacity leaves substantial room for two serialized nodes.
    return DigestOutputContract(max_transport_tokens=max(1, min(int(limit * 0.20), limit // 4)))


def estimate_digest_transport_tokens(digest: DigestNode, token_counter: TokenCounter) -> int:
    value = token_counter.count_payload(digest.model_dump(mode="json"))
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("token counter returned an invalid digest token count")
    return value


def validate_digest_contract(digest: DigestNode, *, contract: DigestOutputContract,
                             token_counter: TokenCounter, stage: str,
                             repair_attempted: bool = False) -> int:
    tokens = estimate_digest_transport_tokens(digest, token_counter)
    reason = None
    if tokens > contract.max_transport_tokens:
        reason = "transport_tokens_exceeded"
    elif len(digest.topics) > contract.max_topics:
        reason = "topic_count_exceeded"
    elif len(digest.key_facts) > contract.max_key_facts:
        reason = "key_fact_count_exceeded"
    if reason:
        raise DigestOutputContractError(
            doc_id=digest.doc_id, stage=stage, response_model=type(digest).__name__,
            estimated_transport_tokens=tokens, contract=contract,
            topic_count=len(digest.topics), key_fact_count=len(digest.key_facts),
            repair_attempted=repair_attempted, reason=reason,
        )
    return tokens
