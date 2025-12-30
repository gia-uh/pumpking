from __future__ import annotations
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from pumpking.models import ChunkNode, DocumentRoot
from pumpking.protocols import StrategyProtocol, ExecutionContext
from pumpking.exceptions import PipelineConfigurationError


def annotate(strategy: StrategyProtocol, alias: Optional[str] = None) -> Tuple[str, StrategyProtocol]:
    """
    Helper function to create an annotation definition for the pipeline syntax.
    """
    final_alias = alias or strategy.__class__.__name__
    return (final_alias, strategy)


class Step:
    """
    Represents a single processing step in the pipeline.
    """
    def __init__(self, strategy: StrategyProtocol, alias: Optional[str] = None) -> None:
        self.strategy = strategy
        self.alias = alias or strategy.__class__.__name__
        self.annotators: Dict[str, StrategyProtocol] = {}

    def __or__(self, annotation: Tuple[str, StrategyProtocol]) -> Step:
        """
        Adds an annotator to this step using the | operator.
        """
        alias, strategy = annotation
        if alias in self.annotators:
            raise PipelineConfigurationError(
                f"Duplicate annotator alias '{alias}' found in step '{self.alias}'."
            )
        self.annotators[alias] = strategy
        return self

    def __rshift__(self, next_step: Union[Step, List[Step]]) -> PumpkingPipeline:
        """
        Starts a pipeline connecting this step to the next using the >> operator.
        """
        return PumpkingPipeline([self, next_step])

    def run(self, initial_input: str) -> DocumentRoot:
        """
        Executes this single step as a pipeline.
        """
        return PumpkingPipeline([self]).run(initial_input)


class PumpkingPipeline:
    """
    Orchestrates the execution of a sequence of Steps.
    """
    def __init__(self, steps: List[Union[Step, List[Step]]] = None) -> None:
        self.steps = steps or []

    def __rshift__(self, next_step: Union[Step, List[Step]]) -> PumpkingPipeline:
        """
        Appends a step or parallel block to the pipeline using the >> operator.
        """
        self.steps.append(next_step)
        return self

    def run(self, initial_input: str) -> DocumentRoot:
        """
        Executes the entire pipeline on the input string.
        """
        root_node = ChunkNode(
            id=str(uuid.uuid4()),
            content=initial_input,
            strategy_label="Input",
            parent_id=None
        )
        
        current_frontier: List[ChunkNode] = [root_node]

        for block in self.steps:
            next_frontier: List[ChunkNode] = []
            
            steps_to_execute = block if isinstance(block, list) else [block]
            
            for parent_node in current_frontier:
                if parent_node.children is None:
                    parent_node.children = []

                for step in steps_to_execute:
                    context = ExecutionContext(annotators=step.annotators) 
                    
                    input_content = parent_node.content if parent_node.content else ""
                    
                    raw_output = step.strategy.execute(input_content, context)
                    
                    output_items = raw_output if isinstance(raw_output, list) else [raw_output]
                    
                    for item in output_items:
                        child_node = step.strategy.to_node(item)
                        
                        child_node.id = str(uuid.uuid4())
                        child_node.parent_id = parent_node.id
                        child_node.strategy_label = step.alias
                        
                        parent_node.children.append(child_node)
                        next_frontier.append(child_node)
            
            if next_frontier:
                current_frontier = next_frontier
        
        return DocumentRoot(
            document=initial_input,
            children=[root_node]
        )