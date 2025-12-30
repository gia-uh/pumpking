from typing import Any, Dict, List, Protocol, runtime_checkable
from pydantic import BaseModel, ConfigDict
from pumpking.models import NERResult, ChunkNode

class ExecutionContext(BaseModel):
    """
    Holds configuration and state for the current execution step.
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
        """
        ...

    def to_node(self, payload: Any) -> ChunkNode:
        """
        Converts a strategy-specific payload into a graph node.
        """
        ...


@runtime_checkable
class NERProviderProtocol(Protocol):
    """
    Protocol for NER providers.
    """
    def extract_entities(self, sentences: List[str], **kwargs: Any) -> List[NERResult]:
        """
        Analyzes a list of sentences and returns entities referencing sentence indices.
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
    def get_document_context(self, document_text: str) -> str:
        """
        Extracts a global semantic summary from the entire document.
        """
        ...

    def get_chunk_context(self, document_context: str, chunk_text: str) -> str:
        """
        Generates the specific situational grounding for a given fragment.
        """
        ...