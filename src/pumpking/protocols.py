from typing import Any, Dict, List, Protocol, runtime_checkable
from pydantic import BaseModel, ConfigDict
from pumpking.models import (
    ChunkPayload, 
    EntityChunkPayload, 
    TopicChunkPayload,
    ZettelChunkPayload,
    ContextualChunkPayload
)

class ExecutionContext(BaseModel):
    """
    Holds configuration and state for the current execution step.
    Defined as a Pydantic model to allow arbitrary types and validation.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    annotators: Dict[str, Any] = {}


@runtime_checkable
class StrategyProtocol(Protocol):
    """
    Protocol defining the interface for all processing strategies.
    """
    def execute(self, data: Any, context: ExecutionContext) -> Any:
        """
        Executes the strategy logic on the provided data.
        Returns raw data or ChunkPayloads, but never ChunkNodes.
        """
        ...


@runtime_checkable
class NERProviderProtocol(Protocol):
    """
    Protocol for NER providers.
    """
    def extract_entities(self, chunks: List[ChunkPayload], **kwargs: Any) -> List[EntityChunkPayload]:
        """
        Analyzes a list of chunk payloads and returns a list of identified entities
        referencing the original source chunks.
        """
        ...


@runtime_checkable
class SummaryProviderProtocol(Protocol):
    """
    Protocol for providers capable of text summarization.
    """
    def summarize(self, chunks: List[ChunkPayload], **kwargs: Any) -> List[ChunkPayload]:
        """
        Analyzes a list of chunks and returns a list of new ChunkPayloads containing 
        the summaries. Each summary payload must reference the original source chunk 
        in its 'children' field.
        """
        ...

@runtime_checkable
class TopicProviderProtocol(Protocol):
    """
    Protocol for providers that assign thematic labels to text blocks.
    """
    def assign_topics(self, chunks: List[ChunkPayload], **kwargs: Any) -> List[TopicChunkPayload]:
        """
        Analyzes a list of chunks, identifies topics, and groups the chunks
        under new TopicChunkPayload objects.
        """
        ...


@runtime_checkable
class ContextualProviderProtocol(Protocol):
    """
    Protocol for providers that generate situational context for fragments.
    """
    def assign_context(self, chunks: List[ChunkPayload], **kwargs: Any) -> List[ContextualChunkPayload]:
        """
        Analyzes a list of chunks and returns a list of ContextualChunkPayloads,
        where each payload enriches the original chunk with situational context.
        """
        ...


@runtime_checkable
class ZettelProviderProtocol(Protocol):
    """
    Defines the contract for providers capable of transforming physical text 
    fragments into atomic knowledge units (Zettels).
    """

    def extract_zettels(self, chunks: List[ChunkPayload], **kwargs: Any) -> List[ZettelChunkPayload]:
        """
        Analyzes a sequence of input chunks to identify and extract atomic 
        concepts.
        """
        ...