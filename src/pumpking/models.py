import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator

class PumpkingBaseModel(BaseModel):
    """
    Base model for all Pumpking objects providing dictionary conversion utilities.
    """
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the model to a dictionary, recursively removing None values and empty collections.
        """
        data = self.model_dump(mode='json', exclude_none=True)
        return self._clean_empty(data)

    @classmethod
    def _clean_empty(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {
                k: v_clean
                for k, v in data.items()
                if (v_clean := cls._clean_empty(v)) not in (None, {}, [])
            }
        elif isinstance(data, list):
            return [
                v_clean
                for v in data
                if (v_clean := cls._clean_empty(v)) not in (None, {}, [])
            ]
        return data


class NERResult(BaseModel):
    """
    Represents a detected entity and the indices of the sentences associated with it.
    """
    entity: str
    label: str
    indices: List[int]


class ChunkPayload(PumpkingBaseModel):
    """
    Transport object returned by Strategies.
    """
    content: Optional[str] = None
    content_raw: Optional[str] = None
    annotations: Dict[str, Any] = Field(default_factory=dict)
    children: Optional[List["ChunkPayload"]] = None


class EntityChunkPayload(ChunkPayload):
    """
    Specialized payload for Entity nodes.
    """
    entity: str
    type: str
    content: Optional[str] = None
    content_raw: Optional[str] = None


class ChunkNode(PumpkingBaseModel):
    """
    A node in the processing graph representing a state in the pipeline history.
    """
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    parent_id: Optional[uuid.UUID] = None
    content: Optional[str] = None
    content_raw: Optional[str] = None
    annotations: Dict[str, Any] = Field(default_factory=dict)
    children: Optional[List["ChunkNode"]] = Field(default_factory=list)

    @model_validator(mode='after')
    def clean_content_raw(self) -> 'ChunkNode':
        """Optimizes storage by removing content_raw if it matches content."""
        if self.content_raw == self.content:
            self.content_raw = None
        return self


class EntityChunkNode(ChunkNode):
    """
    Specialized node for persisting Entity information.
    """
    entity: str
    type: str
    content: Optional[str] = None
    content_raw: Optional[str] = None


class DocumentRoot(PumpkingBaseModel):
    """
    The root container for a processed document tree.
    """
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    document: str
    original_filename: Optional[str] = None
    children: List[ChunkNode] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)