import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator

class PumpkingBaseModel(BaseModel):
    """
    Base abstraction for all data models within the Pumpking architecture.

    This class provides essential utility methods for data serialization and sanitization.
    It ensures that all derived models can be converted to dictionary representations
    free of noise, such as null values or empty data structures, which facilitates
    cleaner storage and API responses.
    """

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the model instance into a dictionary, recursively removing empty values.

        Returns:
            Dict[str, Any]: A sanitized dictionary representation of the model where
            None, empty dictionaries, and empty lists have been pruned.
        """
        data = self.model_dump(mode="json", exclude_none=True)
        return self._clean_empty(data)

    @classmethod
    def _clean_empty(cls, data: Any) -> Any:
        """
        Recursively traverses a data structure to remove empty elements.

        This method filters out None values, empty dictionaries, and empty lists
        from nested structures to ensure a compact representation.

        Args:
            data (Any): The input data structure (dict, list, or primitive).

        Returns:
            Any: The cleaned data structure.
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

    A ChunkPayload encapsulates a segment of text along with its metadata,
    lineage, and annotations. It serves as the standard container that strategies
    receive, process, and produce. By maintaining references to children (source chunks),
    it allows for the construction of a traceable lineage graph from raw text to
    highly processed insights.
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    content: Optional[str] = None
    content_raw: Optional[str] = None
    annotations: Dict[str, Any] = Field(default_factory=dict)
    children: List["ChunkPayload"] = Field(default_factory=list)

    @model_validator(mode="after")
    def clean_fields(self) -> "ChunkPayload":
        """
        Validates and sanitizes the model fields after initialization.

        This validator ensures consistency by enforcing the following rules:
        1. If 'content_raw' is identical to 'content', it is set to None to avoid redundancy.
        2. If 'content' is an empty string, it is set to None.

        Returns:
            ChunkPayload: The validated model instance.
        """
        if self.content_raw == self.content:
            self.content_raw = None
        if self.content == "":
            self.content = None
        return self


class ChunkNode(PumpkingBaseModel):
    """
    Represents a node within the persistent structural graph of the document.

    Unlike ChunkPayload, which is a transient transport container used during
    processing, ChunkNode is designed for graph storage. It wraps a payload
    and explicitly links it to a parent node via a UUID, establishing the
    hierarchical relationship required for tree traversals and reconstruction.
    """

    id: uuid.UUID
    payload: ChunkPayload
    parent_id: Optional[uuid.UUID] = None


class DocumentRoot(PumpkingBaseModel):
    """
    The root anchor for a processed document's graph.

    This class serves as the entry point for a document's structural representation.
    It holds the initial raw content and a unique identifier that connects all
    subsequent nodes generated during the chunking and analysis phases.
    """

    id: uuid.UUID
    document: str


class EntityChunkPayload(ChunkPayload):
    """
    A specialized payload representing a Named Entity identified in the text.

    This class extends ChunkPayload to capture specific attributes of an entity,
    such as its name and classification type (e.g., Person, Organization).
    It retains all standard payload capabilities, including lineage tracking,
    to trace the entity back to its source text.
    """

    entity: str
    type: str


class TopicChunkPayload(ChunkPayload):
    """
    A specialized payload representing a thematic topic or category.

    This class allows for the grouping of content under a unifying semantic label.
    It is used when the processing strategy identifies a dominant subject
    encompassing multiple text segments.
    """

    topic: str


class ContextualChunkPayload(ChunkPayload):
    """
    A specialized payload that enriches a text segment with situational context.

    This model is designed to solve the problem of lost context in retrieval-augmented
    generation (RAG) systems. It stores the explicit 'context' (e.g., resolved pronouns,
    background information) alongside the original content, making the chunk
    self-sufficient for downstream retrieval tasks.
    """

    context: str


class ZettelChunkPayload(ChunkPayload):
    """
    Represents an atomic knowledge unit or 'Zettel' derived from the analysis
    of text fragments.

    Unlike standard physical chunks, a Zettel encapsulates a distinct, synthesized
    concept that possesses its own identity and relationships within a knowledge graph.
    It is the output of deep semantic analysis strategies (Zettelkasten).
    """

    hypothesis: str
    tags: List[str] = Field(default_factory=list)
    related_zettel_ids: List[uuid.UUID] = Field(default_factory=list)