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