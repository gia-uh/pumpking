from typing import Any, Dict, List, Protocol, runtime_checkable
from pydantic import BaseModel, ConfigDict
from pumpking.models import ChunkPayload, ZettelChunkPayload, EntityChunkPayload


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
    Protocol for providers capable of performing Named Entity Recognition.
    """

    def extract_entities(
        self, chunks: List[ChunkPayload], **kwargs: Any
    ) -> List[EntityChunkPayload]:
        """
        Analyzes a list of text chunks and returns a list of identified entities.
        Each EntityChunkPayload must contain the original source chunks in its 'children' field.
        """
        ...


@runtime_checkable
class SummaryProviderProtocol(Protocol):
    """
    Protocol for Summary providers.
    """

    def summarize(self, text: str, **kwargs: Any) -> str:
        """
        Generates a concise summary of the provided text.
        """
        ...


@runtime_checkable
class TopicProviderProtocol(Protocol):
    """
    Protocol for providers that assign thematic labels to text blocks.
    """

    def assign_topics(self, chunks: List[str], **kwargs: Any) -> List[List[str]]:
        """
        Assigns a list of topics to each input fragment.
        """
        ...


@runtime_checkable
class ContextualProviderProtocol(Protocol):
    """
    Protocol for providers that generate situational context for fragments.
    """

    def assign_context(self, chunks: List[str], **kwargs: Any) -> List[str]:
        """
        Assigns a situational context string to each input fragment.
        """
        ...


@runtime_checkable
class ZettelProviderProtocol(Protocol):
    """
    Defines the contract for providers capable of transforming physical text
    fragments into atomic knowledge units (Zettels). Implementations of this
    protocol are responsible for semantic analysis, concept synthesis, and
    the resolution of local graph relationships.
    """

    def extract_zettels(
        self, chunks: List[ChunkPayload], **kwargs: Any
    ) -> List[ZettelChunkPayload]:
        """
        Analyzes a sequence of input chunks to identify and extract atomic
        concepts.

        The provider must ensure that:
        1. Each returned Zettel contains a synthesized 'hypothesis'.
        2. The 'children' field of each Zettel is populated with the relevant
           ChunkPayload objects from the input that serve as evidence.
        3. The 'related_zettel_ids' field is populated to reflect semantic
           relationships identified between the generated Zettels.

        Args:
            chunks: A list of ChunkPayload objects representing the raw
                textual evidence (e.g., paragraphs or sentences).
            **kwargs: Additional configuration parameters for the underlying
                model or extraction logic.

        Returns:
            A list of ZettelChunkPayload objects, each representing a distinct
            concept with its associated metadata and evidence.
        """
        ...
