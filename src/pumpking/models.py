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


class ChunkPayload(PumpkingBaseModel):
    """
    Transport object returned by Strategies.
    
    Attributes:
        content (str): The processed/cleaned text content.
        content_raw (Optional[str]): The original text content before cleaning, if different.
        annotations (Dict[str, Any]): Metadata or analysis results attached to this chunk.
        children (Optional[List[ChunkPayload]]): Nested chunks for hierarchical structures.
    """
    content: str
    content_raw: Optional[str] = None
    annotations: Dict[str, Any] = Field(default_factory=dict)
    children: Optional[List["ChunkPayload"]] = None


class ChunkNode(PumpkingBaseModel):
    """
    A node in the processing graph representing a state in the pipeline history.
    
    Attributes:
        id (uuid.UUID): Unique identifier for the node.
        parent_id (Optional[uuid.UUID]): Reference to the source node.
        content (str): The processed text.
        content_raw (Optional[str]): The original text, stored only if different from content.
        annotations (Dict[str, Any]): Metadata attached during the strategy execution.
        children (Optional[List[ChunkNode]]): Nested nodes representing the tree structure.
    """
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    parent_id: Optional[uuid.UUID] = None
    content: str
    content_raw: Optional[str] = None
    annotations: Dict[str, Any] = Field(default_factory=dict)
    children: Optional[List["ChunkNode"]] = None

    @model_validator(mode='after')
    def clean_content_raw(self) -> 'ChunkNode':
        """Optimizes storage by removing content_raw if it matches content."""
        if self.content_raw == self.content:
            self.content_raw = None
        return self


class DocumentRoot(PumpkingBaseModel):
    """
    The root container for a processed document tree.
    """
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    original_filename: Optional[str] = None
    children: List[ChunkNode] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)