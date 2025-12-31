import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator
class PumpkingBaseModel(BaseModel):
    """
    Base model for all Pumpking components providing recursive dictionary conversion
    and automatic cleanup of empty or null values.
    """
    def to_dict(self) -> Dict[str, Any]:
        data = self.model_dump(mode="json", exclude_none=True)
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
    Data structure representing the identified entity, its category label, 
    and the indices of the sentences where it was detected.
    """
    entity: str
    label: str
    indices: List[int]

class ChunkPayload(PumpkingBaseModel):
    """
    Base data container produced by strategies. It encapsulates the processed text,
    original raw content, and maintains semantic hierarchy through children.
    """
    content: Optional[str] = None
    content_raw: Optional[str] = None
    annotations: Dict[str, Any] = Field(default_factory=dict)
    children: List["ChunkPayload"] = Field(default_factory=list)

    @model_validator(mode="after")
    def clean_content_raw(self) -> "ChunkPayload":
        """
        Removes content_raw if it is identical to content to save space.
        """
        if self.content_raw == self.content:
            self.content_raw = None
        return self

class EntityChunkPayload(ChunkPayload):
    """
    Specialized payload for entity-based strategies containing the entity name 
    and its classification type.
    """
    entity: str
    type: str

class TopicChunkPayload(ChunkPayload):
    """
    Specialized payload for thematic strategies containing the identified topic.
    """
    topic: str

class ContextualChunkPayload(ChunkPayload):
    """
    Specialized payload for chunks enriched with situational or global context.
    """
    context: str

class ChunkNode(PumpkingBaseModel):
    """
    A node in the execution graph. It separates the strategy output (results)
    from the subsequent flow of the pipeline (branches).
    """
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    parent_id: Optional[uuid.UUID] = None
    strategy_label: Optional[str] = None
    results: List[ChunkPayload] = Field(default_factory=list)
    branches: List["ChunkNode"] = Field(default_factory=list)

class EntityChunkNode(ChunkNode):
    """
    Node specialization that preserves entity metadata for direct access 
    within the execution graph.
    """
    entity: Optional[str] = None
    type: Optional[str] = None

class TopicChunkNode(ChunkNode):
    """
    Node specialization that preserves topic metadata for direct access 
    within the execution graph.
    """
    topic: Optional[str] = None

class ContextualChunkNode(ChunkNode):
    """
    Node specialization that preserves contextual metadata for direct access 
    within the execution graph.
    """
    context: Optional[str] = None

class DocumentRoot(PumpkingBaseModel):
    """
    The root of the processed document tree. It holds the original document 
    source and marks the starting points of all execution branches.
    """
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    document: str
    original_filename: Optional[str] = None
    branches: List[ChunkNode] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)