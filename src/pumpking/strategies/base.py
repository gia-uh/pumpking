from abc import ABC, abstractmethod
from typing import Any, List

from pumpking.models import ChunkNode
from pumpking.protocols import StrategyProtocol, ExecutionContext


class BaseStrategy(ABC, StrategyProtocol):
    """
    Abstract base class for all Pumpking strategies.
    Uses standard Python types for strict contract definition.
    """
    SUPPORTED_INPUTS: List[Any] = [str, list[str]]
    PRODUCED_OUTPUT: Any = list[ChunkNode]

    @abstractmethod
    def execute(self, data: Any, context: ExecutionContext) -> Any:
        """
        Execute the strategy logic.

        Args:
            data: Input data matching one of the types in SUPPORTED_INPUTS.
            context: Execution resources.

        Returns:
            Data matching the type defined in PRODUCED_OUTPUT.
        """
        pass

    def _apply_annotators_to_node(self, node: ChunkNode, context: ExecutionContext) -> None:
        """
        Helper: Apply injected annotators to a single node.
        """
        if not context.annotators:
            return

        for alias, strategy in context.annotators.items():
            empty_context = ExecutionContext()
            result = strategy.execute(node.content, empty_context)
            node.annotations[alias] = result

    def _apply_annotators_to_list(self, nodes: List[ChunkNode], context: ExecutionContext) -> None:
        """
        Helper: Apply injected annotators to a list of nodes sequentially.
        """
        if not context.annotators:
            return

        for node in nodes:
            self._apply_annotators_to_node(node, context)