"""Provider-neutral presentation understanding and planning pipeline."""

from .artifacts import write_outline_json
from .pipeline import BatchExtractionError, generate_outline

__all__ = ["BatchExtractionError", "generate_outline", "write_outline_json"]
