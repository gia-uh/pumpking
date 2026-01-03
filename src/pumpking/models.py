import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class PumpkingBaseModel(BaseModel):
    """
    Base abstraction for all Pumpking data models, providing unified serialization behavior.

    This class serves as the foundation for the domain model, ensuring consistent
    conversion to dictionary formats. It implements recursive sanitization logic to
    prune empty values (None, empty dicts, empty lists) from the serialized output,
    resulting in cleaner API payloads and storage representations.
    """

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the model to a dictionary and removes all empty fields recursively.

        Returns:
            Dict[str, Any]: A sanitized dictionary representation of the model instance.
        """
        data = self.model_dump(mode="json", exclude_none=True)
        return self._clean_empty(data)

    @classmethod
    def _clean_empty(cls, data: Any) -> Any:
        """
        Recursively traverses a data structure to prune empty elements.

        Args:
            data: The input structure (dict, list, or primitive) to sanitize.

        Returns:
            The cleaned data structure with None, {}, and [] removed.
        """
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
    The fundamental unit of data transport within the processing pipeline.

    A ChunkPayload acts as a container for a segment of text as it flows through
    strategies. It preserves the content, its raw representation (for debugging or
    formatting), annotations added by the system, and a lineage history via
    child references.
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    content: Optional[str] = None
    content_raw: Optional[str] = None
    annotations: Dict[str, Any] = Field(default_factory=dict)
    children: List["ChunkPayload"] = Field(default_factory=list)

    @model_validator(mode="after")
    def clean_content_raw(self) -> "ChunkPayload":
        """
        Optimizes storage by removing the 'content_raw' field if it is identical
        to the 'content' field.

        Returns:
            ChunkPayload: The validated instance with potentially cleared content_raw.
        """
        if self.content_raw == self.content:
            self.content_raw = None
        return self


class EntityChunkPayload(ChunkPayload):
    """
    A specialized payload representing a named entity extracted from text.

    This class extends the base payload to include strict typing for entity-specific
    metadata, such as the entity's canonical name and its classification type
    (e.g., Person, Organization, Location).
    """

    entity: str
    type: str


class TopicChunkPayload(ChunkPayload):
    """
    A specialized payload representing a thematic cluster or topic.

    Used by topic modeling strategies to encapsulate a high-level subject
    that groups underlying content chunks.
    """

    topic: str


class ContextualChunkPayload(ChunkPayload):
    """
    A specialized payload enriched with situational context.

    Designed for RAG workflows, this payload stores generated context (such as
    resolved pronouns or document-level background) separately from the original
    text content, ensuring the source material remains pristine.
    """

    context: str


class ChunkNode(PumpkingBaseModel):
    """
    Represents a structural node in the execution graph (pipeline trace).

    Unlike a ChunkPayload which holds data, a ChunkNode holds the *result* of a
    strategy execution step and defines the flow to subsequent steps (branches).
    It persists the hierarchy of how data was processed, linking inputs to outputs.
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    parent_id: Optional[uuid.UUID] = None
    strategy_label: Optional[str] = None
    results: List[ChunkPayload] = Field(default_factory=list)
    branches: List["ChunkNode"] = Field(default_factory=list)


class DocumentRoot(PumpkingBaseModel):
    """
    The root anchor of a processed document's execution graph.

    This object encapsulates the initial state of a document before any processing,
    holding the full raw text and metadata. It serves as the origin point for all
    subsequent processing branches.
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    document: str
    original_filename: Optional[str] = None
    branches: List[ChunkNode] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ZettelChunkPayload(ChunkPayload):
    """
    Represents an atomic knowledge unit (Zettel) derived from semantic analysis.

    A Zettel is a synthesized concept with a hypothesis, tags, and relationships
    to other Zettels. It is distinct from a physical text chunk as it represents
    an idea rather than just a segment of characters.

    Attributes:
        hypothesis: The core idea or thesis statement of the Zettel.
        tags: Taxonomic labels for categorization.
        related_zettel_ids: UUIDs of other Zettels connected to this one, forming a graph.
    """

    hypothesis: str
    tags: List[str] = Field(default_factory=list)
    related_zettel_ids: List[uuid.UUID] = Field(default_factory=list)