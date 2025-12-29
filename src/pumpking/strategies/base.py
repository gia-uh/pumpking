from abc import ABC, abstractmethod
from typing import Any, List

from pumpking.models import ChunkPayload
from pumpking.protocols import StrategyProtocol, ExecutionContext


class BaseStrategy(ABC, StrategyProtocol):
    SUPPORTED_INPUTS: List[Any] = [str, list[str]]
    PRODUCED_OUTPUT: Any = List[ChunkPayload]

    @abstractmethod
    def execute(self, data: Any, context: ExecutionContext) -> Any:
        pass

    def _apply_annotators_to_payload(self, content: str, context: ExecutionContext) -> ChunkPayload:
        payload = ChunkPayload(content=content)
        
        if not context.annotators:
            return payload

        for alias, strategy in context.annotators.items():
            empty_context = ExecutionContext()
            result = strategy.execute(content, empty_context)
            payload.annotations[alias] = result
        
        return payload

    def _apply_annotators_to_list(self, items: List[str], context: ExecutionContext) -> List[ChunkPayload]:
        return [self._apply_annotators_to_payload(item, context) for item in items]