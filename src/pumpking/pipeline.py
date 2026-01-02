import uuid
import inspect
from typing import (
    List, 
    Union, 
    Any, 
    Optional, 
    Type, 
    Dict, 
    get_type_hints, 
    get_origin, 
    get_args
)
from pumpking.models import ChunkNode, DocumentRoot, ChunkPayload
from pumpking.protocols import ExecutionContext, StrategyProtocol
from pumpking.exceptions import PipelineConfigurationError

class Step:
    """
    Represents an atomic processing stage within the document graph.

    This class encapsulates a specific processing strategy and manages the flow
    of data between graph nodes. Crucially, it acts as an intelligent dispatcher
    that adapts its execution mode based on the capabilities of the underlying
    strategy.

    By inspecting the type signatures of the strategy at runtime, this class
    decides whether to process input items sequentially (Iterative/FlatMap) or
    to pass them as a comprehensive group (Batching). This enables optimization
    opportunities, such as reducing LLM API calls, without imposing complexity
    on the pipeline configuration.
    """

    def __init__(self, strategy: StrategyProtocol, alias: Optional[str] = None) -> None:
        self.strategy = strategy
        self.alias = alias or strategy.__class__.__name__
        self.annotators: Dict[str, StrategyProtocol] = {}

    def execute(
        self, 
        input_nodes: List[Union[DocumentRoot, ChunkNode]], 
        context: ExecutionContext
    ) -> List[ChunkNode]:
        """
        Executes the strategy logic across the provided frontier of nodes.

        This method serves as the main entry point for data processing in this step.
        It aggregates data from input nodes and dispatches it to the strategy.

        Two execution modes are supported:
        1. Batch Mode: If the strategy accepts a List, all data is passed at once.
           Lineage (Parent-Child relationship) MUST be preserved by the strategy
           setting the 'children' field in the result payloads.
        2. Iterative Mode: If the strategy accepts single items, the pipeline
           iterates through inputs, calls the strategy for each, and manually
           assigns the parent ID based on the input node being processed.

        Args:
            input_nodes: The list of nodes from the previous step (or the root).
            context: The shared execution context for the pipeline.

        Returns:
            A list of newly created ChunkNodes resulting from the execution.
        """
        new_nodes = []
        
        # Tuple of (Data Item, Parent Node ID)
        inputs_with_parents = []
        
        # 1. Extract Data and Map Lineage
        for node in input_nodes:
            p_id = node.id
            if isinstance(node, DocumentRoot):
                inputs_with_parents.append((node.document, p_id))
            else:
                for payload in node.results:
                    inputs_with_parents.append((payload, p_id))

        if not inputs_with_parents:
            return []

        context.annotators = self.annotators

        # 2. Dispatch Execution based on Strategy Capabilities
        if self._supports_batch_input(self.strategy):
            new_nodes = self._execute_batch(inputs_with_parents, input_nodes, context)
        else:
            new_nodes = self._execute_iterative(inputs_with_parents, input_nodes, context)
            
        return new_nodes

    def _execute_batch(
        self, 
        inputs_with_parents: List[tuple], 
        origin_nodes: List[Union[DocumentRoot, ChunkNode]],
        context: ExecutionContext
    ) -> List[ChunkNode]:
        """
        Handles strategies that support batch processing (List input).
        Rely on 'children' field in payloads to resolve parents.
        """
        all_inputs = [item for item, _ in inputs_with_parents]
        lineage_map = {id(item): pid for item, pid in inputs_with_parents}
        
        raw_results = self.strategy.execute(all_inputs, context)
        
        # Normalize result to list
        if not isinstance(raw_results, list):
            raw_results = [raw_results]
            
        new_nodes = []
        for result_payload in raw_results:
            # For batch strategies, we expect them to link back to source via children
            parent_id = self._resolve_parent_id(result_payload, lineage_map)
            
            # Create and attach node
            node = self._create_node([result_payload], parent_id=parent_id)
            self._attach_to_parent(node, parent_id, origin_nodes)
            new_nodes.append(node)
            
        return new_nodes

    def _execute_iterative(
        self, 
        inputs_with_parents: List[tuple], 
        origin_nodes: List[Union[DocumentRoot, ChunkNode]],
        context: ExecutionContext
    ) -> List[ChunkNode]:
        """
        Handles strategies that process items one by one.
        Manually tracks parent ID from the input loop.
        """
        new_nodes = []
        for item, p_id in inputs_with_parents:
            raw_result = self.strategy.execute(item, context)
            
            # Normalize result to list (Strategy might return one or many payloads)
            if isinstance(raw_result, list):
                results_list = raw_result
            else:
                results_list = [raw_result]
            
            for res in results_list:
                # In iterative mode, we know exactly which parent this result belongs to
                node = self._create_node([res], parent_id=p_id)
                self._attach_to_parent(node, p_id, origin_nodes)
                new_nodes.append(node)
                
        return new_nodes

    def _attach_to_parent(
        self, 
        child_node: ChunkNode, 
        parent_id: Optional[uuid.UUID], 
        origin_nodes: List[Union[DocumentRoot, ChunkNode]]
    ) -> None:
        """Helper to find the parent node object and append the child."""
        if parent_id is None:
            return
        for origin in origin_nodes:
            if origin.id == parent_id:
                origin.branches.append(child_node)
                break

    def _supports_batch_input(self, strategy: StrategyProtocol) -> bool:
        """
        Determines if the strategy's execute method accepts a List as input.
        inspects parameters robustly, ignoring 'self' and 'context'.
        """
        try:
            sig = inspect.signature(strategy.execute)
            type_hints = get_type_hints(strategy.execute)
            
            for name, param in sig.parameters.items():
                # Skip implicit self and the context argument
                if name == 'self' or name == 'context':
                    continue
                
                # Check the first data argument found
                # We prioritize the resolved type hint if available
                hint = type_hints.get(name, param.annotation)
                return self._type_allows_list(hint)
                
            return False # No data argument found?
            
        except Exception:
            # Fallback: strict safety, assume no batch support if introspection fails
            return False

    def _type_allows_list(self, type_hint: Any) -> bool:
        """
        Recursively checks if a type hint implies support for List input.
        """
        if type_hint is Any:
            # Ideally Any allows list, but for safety in ambiguous mocks
            # checking if it STRICTLY allows list is safer. 
            # However, 'Any' implies it can handle anything.
            # Let's assume True, but Mocks should be explicit.
            return True

        origin = get_origin(type_hint)
        
        if origin is list or origin is List:
            return True
        if type_hint is list or type_hint is List:
            return True

        if origin is Union:
            return any(self._type_allows_list(arg) for arg in get_args(type_hint))

        return False

    def _resolve_parent_id(
        self, 
        payload: ChunkPayload, 
        lineage_map: Dict[int, uuid.UUID]
    ) -> Optional[uuid.UUID]:
        """
        Identifies the parent node ID for a given result payload based on lineage.
        """
        if payload.children:
            primary_source = payload.children[0]
            source_id = id(primary_source)
            if source_id in lineage_map:
                return lineage_map[source_id]
        return None

    def _get_node_class(self, payloads: List[ChunkPayload]) -> Type[ChunkNode]:
        """Dynamically identifies the appropriate ChunkNode subclass."""
        if payloads and hasattr(payloads[0], "__node_class__"):
            return getattr(payloads[0], "__node_class__")
        return ChunkNode

    def _create_node(
        self, 
        payloads: List[ChunkPayload], 
        parent_id: Optional[uuid.UUID] = None
    ) -> ChunkNode:
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

    def __rshift__(self, next_val: Union['Step', List[Any]]) -> 'PumpkingPipeline':
        """
        Defines sequential topology. 
        Returns a list structure [self, next] to be parsed by the Pipeline.
        """
        return [self, next_val]

    def __or__(self, annotator_step: 'Step') -> 'Step':
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

    def __init__(self, structure: Union[Step, List[Any]]) -> None:
        self.steps = self._initialize_topology(structure)

    def _initialize_topology(self, structure: Any) -> List[Any]:
        if isinstance(structure, Step):
            return [structure]
        
        final_steps = []
        if isinstance(structure, list):
            for item in structure:
                final_steps.append(item)
        return final_steps

    def run(
        self, 
        input_data: Union[str, DocumentRoot], 
        filename: Optional[str] = None
    ) -> DocumentRoot:
        """
        Main execution loop.
        """
        if isinstance(input_data, str):
            root = DocumentRoot(document=input_data, original_filename=filename)
        else:
            root = input_data

        current_context = ExecutionContext()
        current_frontier: List[Union[DocumentRoot, ChunkNode]] = [root]

        for step_item in self.steps:
            next_frontier = []
            
            if isinstance(step_item, list):
                # Parallel branches
                for sub_step in step_item:
                    next_frontier.extend(sub_step.execute(current_frontier, current_context))
            else:
                # Sequential step
                next_frontier.extend(step_item.execute(current_frontier, current_context))
            
            current_frontier = next_frontier
            
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