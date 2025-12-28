from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union

from pumpking.protocols import StrategyProtocol


def annotate(strategy: StrategyProtocol, alias: Optional[str] = None) -> Tuple[str, StrategyProtocol]:
    """
    Helper function to create an annotation tuple for the DSL.

    Args:
        strategy: The strategy instance to execute.
        alias: Optional name. If None, strategy.__class__.__name__ is used.

    Returns:
        A tuple (alias, strategy) compatible with the '|' operator.
    """
    final_alias = alias or strategy.__class__.__name__
    return (final_alias, strategy)


class Step:
    """
    Wrapper around a strategy to enable DSL operators for pipelining and annotation.
    """
    def __init__(self, strategy: StrategyProtocol, alias: Optional[str] = None) -> None:
        """
        Initialize the step wrapper.

        Args:
            strategy: The core strategy instance to be executed in this step.
            alias: Optional identifier. Defaults to the strategy class name.
                   Useful for resolving naming collisions in parallel blocks.
        """
        self.strategy = strategy
        self.alias = alias or strategy.__class__.__name__
        self.annotators: Dict[str, StrategyProtocol] = {}

    def __or__(self, annotation: Tuple[str, StrategyProtocol]) -> Step:
        """
        Register an annotator using the '|' operator.

        Args:
            annotation: A tuple (alias, strategy) from the 'annotate' helper.

        Returns:
            self: Returns the same Step instance to allow chaining.
        """
        alias, strategy = annotation
        self.annotators[alias] = strategy
        return self

    def __rshift__(self, next_step: Union[Step, List[Step]]) -> PumpkingPipeline:
        """
        Connect this step to the next one (or list) using the '>>' operator.

        Args:
            next_step: The next Step instance or list of Steps.

        Returns:
            PumpkingPipeline: A new pipeline containing the sequence.
        """
        return PumpkingPipeline([self, next_step])


class PumpkingPipeline:
    """
    Represents a sequence of processing steps defined via the DSL.
    """
    def __init__(self, steps: List[Union[Step, List[Step]]] = None) -> None:
        """
        Initialize the pipeline.

        Args:
            steps: Initial list of steps.
        """
        self.steps = steps or []

    def __rshift__(self, next_step: Union[Step, List[Step]]) -> PumpkingPipeline:
        """
        Append a step (or parallel block) to the existing pipeline.

        Args:
            next_step: The Step instance or list of Steps to append.

        Returns:
            self: Returns the pipeline instance to allow chaining.
        """
        self.steps.append(next_step)
        return self