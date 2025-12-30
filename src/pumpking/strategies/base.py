from typing import Any, List, Optional
from pumpking.models import ChunkPayload, ChunkNode
from pumpking.protocols import StrategyProtocol, ExecutionContext

class BaseStrategy(StrategyProtocol):
    """
    Base class for all strategies, providing default implementations.
    """
    # Essential attributes required by validation tests
    SUPPORTED_INPUTS: List[Any] = [str, list[str]]
    PRODUCED_OUTPUT: Any = List[ChunkPayload]

    def execute(self, data: Any, context: ExecutionContext) -> Any:
        """
        Abstract method to be implemented by concrete strategies.
        """
        raise NotImplementedError("Strategies must implement execute method.")

    def to_node(self, payload: Any) -> ChunkNode:
        """
        Default implementation to convert a ChunkPayload into a ChunkNode.
        Recursively converts children payloads if present.
        """
        if isinstance(payload, ChunkPayload):
            children_nodes = []
            if payload.children:
                children_nodes = [self.to_node(child) for child in payload.children]

            return ChunkNode(
                content=payload.content,
                content_raw=payload.content_raw,
                annotations=payload.annotations,
                children=children_nodes
            )
        
        return ChunkNode(content=str(payload))

    def _apply_annotators_to_payload(
        self, 
        content: str, 
        context: ExecutionContext, 
        content_raw: str = None
    ) -> ChunkPayload:
        """
        Helper method to apply configured annotators to the generated content.
        """
        payload = ChunkPayload(
            content=content,
            content_raw=content_raw or content
        )
        
        if not context.annotators:
            return payload

        for alias, annotator in context.annotators.items():
            annotation_result = annotator.execute(content, ExecutionContext())
            payload.annotations[alias] = annotation_result
            
        return payload

    def _apply_annotators_to_list(self, items: List[str], context: ExecutionContext) -> List[ChunkPayload]:
        """
        Helper to convert a simple list of strings into annotated ChunkPayloads.
        Required by Mock strategies in tests.
        """
        return [self._apply_annotators_to_payload(item, context) for item in items]