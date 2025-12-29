from __future__ import annotations
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from pumpking.models import ChunkNode, ChunkPayload
from pumpking.protocols import StrategyProtocol, ExecutionContext
from pumpking.exceptions import PipelineConfigurationError


def annotate(strategy: StrategyProtocol, alias: Optional[str] = None) -> Tuple[str, StrategyProtocol]:
    final_alias = alias or strategy.__class__.__name__
    return (final_alias, strategy)


class Step:
    def __init__(self, strategy: StrategyProtocol, alias: Optional[str] = None) -> None:
        self.strategy = strategy
        self.alias = alias or strategy.__class__.__name__
        self.annotators: Dict[str, StrategyProtocol] = {}

    def __or__(self, annotation: Tuple[str, StrategyProtocol]) -> Step:
        alias, strategy = annotation
        if alias in self.annotators:
            raise PipelineConfigurationError(
                f"Duplicate annotator alias '{alias}' found in step '{self.alias}'. "
                "Annotator aliases must be unique within a step."
            )
        self.annotators[alias] = strategy
        return self

    def __rshift__(self, next_step: Union[Step, List[Step]]) -> PumpkingPipeline:
        return PumpkingPipeline([self, next_step])

    def run(self, initial_input: str) -> ChunkNode:
        return PumpkingPipeline([self]).run(initial_input)


class PumpkingPipeline:
    def __init__(self, steps: List[Union[Step, List[Step]]] = None) -> None:
        self.steps = steps or []

    def __rshift__(self, next_step: Union[Step, List[Step]]) -> PumpkingPipeline:
        self.steps.append(next_step)
        return self

    def run(self, initial_input: str) -> ChunkNode:
        """
        Executes the pipeline, building a hierarchical tree of ChunkNodes.
        
        Logic:
        1. Pipeline creates Root Node.
        2. Pipeline passes parent content to Strategy.
        3. Strategy returns ChunkPayloads (content + annotations).
        4. Pipeline converts Payloads to ChunkNodes (assigns IDs, parent_ids).
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
                    # Context carries annotators to the strategy
                    context = ExecutionContext(annotators=step.annotators) 
                    
                    # Execute Strategy (Pure Processing)
                    result = step.strategy.execute(parent_node.content, context)
                    
                    # Normalize to list
                    output_items = result if isinstance(result, list) else [result]
                    
                    # Pipeline acts as Graph Builder
                    for item in output_items:
                        content_str = ""
                        annotations = {}
                        
                        if isinstance(item, ChunkPayload):
                            content_str = item.content
                            annotations = item.annotations
                        else:
                            # Fallback for strategies returning raw strings
                            content_str = str(item)
                        
                        child_node = ChunkNode(
                            id=str(uuid.uuid4()),
                            content=content_str,
                            parent_id=parent_node.id,
                            annotations=annotations
                        )
                        
                        next_frontier.append(child_node)
            
            if next_frontier:
                current_frontier = next_frontier
        
        return root_node