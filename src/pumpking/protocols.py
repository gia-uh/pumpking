from __future__ import annotations
from typing import Any, Dict, List, Protocol, runtime_checkable
from dataclasses import dataclass, field

@dataclass
class ExecutionContext:
    """
    Carries the execution resources and configuration down the pipeline.

    Attributes:
        annotators: A dictionary mapping alias strings to initialized strategy instances.
    """
    annotators: Dict[str, StrategyProtocol] = field(default_factory=dict)


@runtime_checkable
class StrategyProtocol(Protocol):
    """
    Strict interface that all processing strategies must implement.
    """
    SUPPORTED_INPUTS: List[Any]
    PRODUCED_OUTPUT: Any

    def execute(self, data: Any, context: ExecutionContext) -> Any:
        """
        Process the input data using the provided context.

        Args:
            data: The input data.
            context: The execution context.

        Returns:
            The processed output.
        """
        ...