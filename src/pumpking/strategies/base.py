from abc import ABC, abstractmethod
from typing import Any, List, Optional

from pumpking.models import ChunkPayload
from pumpking.protocols import StrategyProtocol, ExecutionContext


class BaseStrategy(ABC, StrategyProtocol):
    """
    Abstract base class for all processing strategies.
    
    Provides helper methods for handling annotators and payload creation.
    """
    SUPPORTED_INPUTS: List[Any] = [str, list[str]]
    PRODUCED_OUTPUT: Any = List[ChunkPayload]

    @abstractmethod
    def execute(self, data: Any, context: ExecutionContext) -> Any:
        """
        Executes the strategy logic.
        
        Args:
            data (Any): The input data (usually string or list of strings).
            context (ExecutionContext): Context containing configured annotators.
            
        Returns:
            Any: Typically a List[ChunkPayload] or a primitive type.
        """
        pass

    def _apply_annotators_to_payload(
        self, 
        content: str, 
        context: ExecutionContext, 
        content_raw: Optional[str] = None
    ) -> ChunkPayload:
        """
        Wraps content in a ChunkPayload and executes any configured annotators.
        
        Args:
            content (str): The cleaned content.
            context (ExecutionContext): Context containing annotators to run.
            content_raw (Optional[str]): Original raw content.
            
        Returns:
            ChunkPayload: The constructed payload with annotations.
        """
        payload = ChunkPayload(content=content, content_raw=content_raw)
        
        if not context.annotators:
            return payload

        for alias, strategy in context.annotators.items():
            empty_context = ExecutionContext()
            result = strategy.execute(content, empty_context)
            payload.annotations[alias] = result
        
        return payload

    def _apply_annotators_to_list(self, items: List[str], context: ExecutionContext) -> List[ChunkPayload]:
        """
        Helper to convert a simple list of strings into annotated ChunkPayloads.
        Assumes raw content is identical to processed content.
        """
        return [self._apply_annotators_to_payload(item, context) for item in items]