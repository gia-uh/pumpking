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
    Represents the shared state and configuration environment for a single execution 
    lifecycle within the pipeline.

    This class serves as a carrier for runtime dependencies, such as annotators 
    or global settings, that need to be accessible across different strategies 
    without being tightly coupled to their constructors. It uses Pydantic 
    configuration to allow arbitrary types, enabling the storage of complex 
    objects like strategy instances or database connections.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    annotators: Dict[str, Any] = {}


@runtime_checkable
class StrategyProtocol(Protocol):
    """
    Defines the fundamental interface for all processing strategies within the architecture.

    Any class attempting to process data—whether it is a simple text splitter, 
    a complex recursive chunker, or an annotator—must adhere to this contract. 
    This uniformity allows strategies to be composed, chained, or nested 
    interchangeably within the pipeline.
    """
    def execute(self, data: Any, context: ExecutionContext) -> Any:
        """
        Performs the core processing logic of the strategy.

        Args:
            data: The input to be processed. This can be raw text, a ChunkPayload, 
                  or a list of payloads, depending on the strategy's specialization.
            context: The shared execution environment containing runtime dependencies 
                     and configuration.

        Returns:
            The result of the processing. This is typically a list of ChunkPayloads, 
            but can be a single payload or raw data in intermediate steps. 
            It should not return ChunkNodes, as those are reserved for the 
            persistence layer of the pipeline.
        """
        ...


@runtime_checkable
class NERProviderProtocol(Protocol):
    """
    Defines the contract for providers capable of Named Entity Recognition (NER).

    Classes implementing this protocol bridge the gap between the internal 
    ChunkPayload architecture and external NER engines (e.g., LLMs, SpaCy). 
    They are responsible for identifying entities within text segments and 
    returning them as specialized payloads that link back to their source.
    """
    def extract_entities(self, chunks: List[ChunkPayload], **kwargs: Any) -> List[EntityChunkPayload]:
        """
        Analyzes a batch of chunks to identify and extract named entities.

        Args:
            chunks: A list of source payloads containing the text to be analyzed.
            **kwargs: Additional configuration parameters specific to the provider 
                      (e.g., model selection, confidence thresholds).

        Returns:
            A list of EntityChunkPayload objects. Each object represents a unique 
            entity occurrence and must contain references to the original source 
            chunks in its 'children' field to preserve lineage.
        """
        ...


@runtime_checkable
class SummaryProviderProtocol(Protocol):
    """
    Defines the contract for providers specialized in text summarization.

    Implementations of this protocol are responsible for distilling the information 
    contained in one or multiple chunks into a concise summary. This is often 
    used in hierarchical chunking strategies where higher-level nodes summarize 
    their children.
    """
    def summarize(self, chunks: List[ChunkPayload], **kwargs: Any) -> List[ChunkPayload]:
        """
        Generates summaries for the provided chunks.

        Args:
            chunks: A list of source payloads to be summarized. Depending on the 
                    implementation, this could result in one summary per chunk 
                    or an aggregated summary for the entire batch.
            **kwargs: Additional configuration parameters specific to the provider.

        Returns:
            A list of ChunkPayloads containing the generated summaries. Each 
            summary payload must explicitly link back to the source chunks it 
            summarizes via the 'children' field.
        """
        ...

@runtime_checkable
class TopicProviderProtocol(Protocol):
    """
    Defines the contract for providers capable of Topic Modeling and Classification.

    Implementations are tasked with analyzing text segments to identify dominant 
    themes or topics. This protocol supports workflows where content is grouped 
    semantically rather than structurally.
    """
    def assign_topics(self, chunks: List[ChunkPayload], **kwargs: Any) -> List[TopicChunkPayload]:
        """
        Analyzes a batch of chunks and categorizes them into topics.

        Args:
            chunks: A list of source payloads to be classified.
            **kwargs: Additional parameters, such as a predefined taxonomy or 
                      instructions for dynamic topic discovery.

        Returns:
            A list of TopicChunkPayload objects. These payloads represent the 
            topics themselves and contain the original content chunks as children, 
            effectively inverting the relationship (Topic -> Content).
        """
        ...


@runtime_checkable
class ContextualProviderProtocol(Protocol):
    """
    Defines the contract for providers that generate situational context.

    This protocol is critical for Retrieval-Augmented Generation (RAG) workflows 
    where isolated text chunks lose meaning. Implementations analyze the 
    chunk within its broader document window to generate metadata (context) 
    that clarifies ambiguities, such as pronoun references or implicit subjects.
    """
    def assign_context(self, chunks: List[ChunkPayload], **kwargs: Any) -> List[ContextualChunkPayload]:
        """
        Enriches a batch of chunks with generated situational context.

        Args:
            chunks: A list of source payloads that require contextualization.
            **kwargs: Additional parameters for the provider.

        Returns:
            A list of ContextualChunkPayloads. Each payload wraps the original 
            content and adds a separate 'context' field, ensuring that the 
            original text remains unaltered while being semantically enriched.
        """
        ...


@runtime_checkable
class ZettelProviderProtocol(Protocol):
    """
    Defines the contract for providers implementing the Zettelkasten method.

    Implementations of this protocol must be capable of deep semantic analysis, 
    transforming raw text into "Zettels"—atomic, self-contained units of knowledge. 
    This involves synthesizing hypotheses, extracting evidence, and identifying 
    relationships between concepts.
    """

    def extract_zettels(self, chunks: List[ChunkPayload], **kwargs: Any) -> List[ZettelChunkPayload]:
        """
        Analyzes input chunks to extract atomic knowledge units.

        Args:
            chunks: A list of source payloads serving as the source material.
            **kwargs: Configuration parameters, potentially including context 
                      from previously extracted Zettels to support linking.

        Returns:
            A list of ZettelChunkPayload objects. Each Zettel represents a 
            synthesized concept with its own identity, linked to the source 
            text via the 'children' field and potentially linked to other 
            Zettels via IDs.
        """
        ...