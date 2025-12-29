from typing import Any, Dict, List, Protocol, runtime_checkable
from pydantic import BaseModel, ConfigDict

from pumpking.models import NERResult


@runtime_checkable
class NERProviderProtocol(Protocol):
    """
    Protocol for NER providers.
    """
    def analyze(self, sentences: List[str]) -> List[NERResult]:
        """
        Analyzes a list of sentences and returns entities referencing sentence indices.
        
        Args:
            sentences: The list of text segments to analyze.
            
        Returns:
            List[NERResult]: Entities with the indices of the sentences they contain.
        """
        ...


@runtime_checkable
class StrategyProtocol(Protocol):
    """
    Protocol defining the interface for all processing strategies.
    """
    def execute(self, data: Any, context: 'ExecutionContext') -> Any:
        """
        Executes the strategy on the provided data.
        """
        ...


class ExecutionContext(BaseModel):
    """
    Holds configuration and state for the current execution step.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    annotators: Dict[str, StrategyProtocol] = {}