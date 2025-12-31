import uuid
from typing import List, Union, Any, Optional, Type
from pumpking.models import ChunkNode, DocumentRoot, ChunkPayload
from pumpking.protocols import ExecutionContext
from pumpking.exceptions import PipelineConfigurationError

class Step:
    """
    Represents an atomic processing stage within the document graph.
    """

    def __init__(self, strategy: Any, alias: Optional[str] = None):
        self.strategy = strategy
        self.alias = alias or strategy.__class__.__name__
        self.annotators = {}

    def _get_node_class(self, payloads: List[ChunkPayload]) -> Type[ChunkNode]:
        """Dynamically identifies the appropriate ChunkNode subclass."""
        if payloads and hasattr(payloads[0], "__node_class__"):
            return getattr(payloads[0], "__node_class__")
        return ChunkNode

    def _create_node(self, payloads: List[ChunkPayload], parent_id: Optional[uuid.UUID] = None) -> ChunkNode:
        """Instantiates a node and maps specialized attributes from the payload."""
        node_class = self._get_node_class(payloads)
        node_data = {
            "parent_id": parent_id,
            "strategy_label": self.alias,
            "results": payloads
        }

        if payloads and node_class is not ChunkNode:
            for field in node_class.model_fields:
                if field not in node_data and hasattr(payloads[0], field):
                    node_data[field] = getattr(payloads[0], field)

        return node_class(**node_data)

    def execute(self, input_node: Union[DocumentRoot, ChunkNode], context: ExecutionContext) -> List[ChunkNode]:
        """
        Executes the step logic with Inversion of Control for annotations.
        """
        new_nodes = []
        p_id = input_node.id
        
        inputs = [input_node.document] if isinstance(input_node, DocumentRoot) else [r.content for r in input_node.results]

        context.annotators = self.annotators

        for data in inputs:
            raw_result = self.strategy.execute(data, context)
            
            if not isinstance(raw_result, list):
                raw_result = [raw_result]
            
            normalized_payloads = []
            for item in raw_result:
                if isinstance(item, ChunkPayload):
                    normalized_payloads.append(item)
                else:
                    normalized_payloads.append(ChunkPayload(content=str(item), content_raw=str(item)))

            node = self._create_node(normalized_payloads, parent_id=p_id)
            input_node.branches.append(node)
            new_nodes.append(node)
            
        return new_nodes

    def __rshift__(self, next_val: Any) -> List[Any]:
        """
        Defines sequential topology. 
        Returns a list structure [self, next] to be parsed by the Pipeline.
        """
        return [self, next_val]

    def __or__(self, annotator_step: Any) -> 'Step':
        """
        Attaches a local annotator to this step.
        Raises PipelineConfigurationError if the alias is already in use.
        """
        if annotator_step.alias in self.annotators:
            raise PipelineConfigurationError(f"Duplicate annotator alias '{annotator_step.alias}'")
            
        self.annotators[annotator_step.alias] = annotator_step.strategy
        return self

class PumpkingPipeline:
    """
    Orchestrates the execution flow (Sequential & Parallel).
    """

    def __init__(self, structure: Union[Step, List[Any]]):
        self.steps = self._initialize_topology(structure)

    def _initialize_topology(self, structure: Any) -> List[Any]:
        """Normalizes the topology structure."""
        if isinstance(structure, Step):
            return [structure]
        
        final_steps = []
        if isinstance(structure, list):
            for item in structure:
                final_steps.append(item)
        return final_steps

    def run(self, input_data: Union[str, DocumentRoot], filename: Optional[str] = None) -> DocumentRoot:
        """
        Main execution loop.
        """
        if isinstance(input_data, str):
            root = DocumentRoot(document=input_data, original_filename=filename)
        else:
            root = input_data

        current_context = ExecutionContext()
        current_nodes = [root]

        for step_item in self.steps:
            next_frontier = []
            
            if isinstance(step_item, list):
                for sub_step in step_item:
                    for node in current_nodes:
                        next_frontier.extend(sub_step.execute(node, current_context))
            else:
                for node in current_nodes:
                    next_frontier.extend(step_item.execute(node, current_context))
            
            current_nodes = next_frontier
            
        return root

    def __rshift__(self, next_val: Union[Step, List[Any]]) -> 'PumpkingPipeline':
        """
        Allows appending steps to an instantiated pipeline.
        Enables the syntax: PumpkingPipeline(...) >> step
        """
        if isinstance(next_val, Step):
            self.steps.append(next_val)
        elif isinstance(next_val, list):
            self.steps.append(next_val)
        else:
            raise PipelineConfigurationError(f"Invalid type for pipeline chaining: {type(next_val)}")
            
        return self

def annotate(strategy: Any, alias: str) -> Step:
    """Helper to create annotation steps."""
    return Step(strategy, alias=alias)