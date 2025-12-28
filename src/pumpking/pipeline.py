from __future__ import annotations
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from pumpking.models import ChunkNode
from pumpking.protocols import StrategyProtocol, ExecutionContext
from pumpking.exceptions import PipelineConfigurationError


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
        
        Raises:
            PipelineConfigurationError: If an annotator with the same alias already exists.
        """
        alias, strategy = annotation
        if alias in self.annotators:
            raise PipelineConfigurationError(
                f"Duplicate annotator alias '{alias}' found in step '{self.alias}'. "
                "Annotator aliases must be unique within a step."
            )
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

    def run(self, initial_input: str) -> ChunkNode:
        """
        Convenience method to execute this single step as if it were a pipeline.
        
        This allows executing atomic steps without manually wrapping them 
        in a PumpkingPipeline container.

        Args:
            initial_input: The raw string content to start the execution.

        Returns:
            The Root ChunkNode resulting from this single step's execution.
        """
        # Self-wrap in a temporary pipeline to reuse the execution logic
        return PumpkingPipeline([self]).run(initial_input)


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

    def run(self, initial_input: str) -> ChunkNode:
        """
        Executes the pipeline, building a hierarchical tree of ChunkNodes.

        The pipeline manages the context:
        1. Creates a Root Node with initial_input.
        2. Iterates through steps. Each step reads from the current frontier of nodes.
        3. Strategies process content -> Pipeline converts output to NEW child nodes.
        4. Annotators run on the NEW child nodes.
        5. The frontier advances to these new nodes.

        Args:
            initial_input: The raw string content to start the pipeline.

        Returns:
            The Root ChunkNode.
        """
        root_node = ChunkNode(
            id=str(uuid.uuid4()),
            content=initial_input,
            parent_id=None
        )
        
        current_frontier: List[ChunkNode] = [root_node]

        for block in self.steps:
            next_frontier: List[ChunkNode] = []
            
            steps_to_execute = block if isinstance(block, list) else [block]
            
            for parent_node in current_frontier:
                for step in steps_to_execute:
                    context = ExecutionContext() 
                    raw_output = step.strategy.execute(parent_node.content, context)
                    
                    output_items = raw_output if isinstance(raw_output, list) else [raw_output]
                    
                    for item in output_items:
                        content_str = str(item)
                        
                        child_node = ChunkNode(
                            id=str(uuid.uuid4()),
                            content=content_str,
                            parent_id=parent_node.id
                        )
                        
                        self._apply_step_annotators(step, child_node)
                        
                        next_frontier.append(child_node)
            
            if next_frontier:
                current_frontier = next_frontier
        
        return root_node

    def _apply_step_annotators(self, step: Step, node: ChunkNode) -> None:
        """
        Executes the step's annotators on the newly created node.
        """
        if not step.annotators:
            return

        for alias, annotator in step.annotators.items():
            context = ExecutionContext()
            annotation_result = annotator.execute(node.content, context)
            node.annotations[alias] = annotation_result