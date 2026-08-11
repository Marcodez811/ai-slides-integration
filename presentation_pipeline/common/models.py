"""Base model for all presentation-pipeline contracts."""

from pydantic import BaseModel, ConfigDict


class PipelineModel(BaseModel):
    """Strict base model that prevents unnoticed contract drift."""

    model_config = ConfigDict(extra="forbid")
