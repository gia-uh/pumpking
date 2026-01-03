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
    Represents an atomic processing stage within the document execution graph.

    This class serves as a smart wrapper around a specific processing strategy. 
    Its primary responsibility is to manage the flow of data between graph nodes, 
    acting as an intelligent dispatcher that adapts its execution mode based on 
    the capabilities of the underlying strategy.

    Key capabilities include:
    1. Runtime Introspection: It inspects the type signatures of the strategy's 
       'execute' method to determine if it supports batch processing (List input) 
       or requires iterative processing (single item input).
    2. Execution Mode Adaptation: Based on introspection, it automatically switches 
       between 'Batch Mode' (passing all inputs at once for optimization, e.g., 
       LLM API calls) and 'Iterative Mode' (processing items one by one).
    3. Graph Construction: It handles the creation of new ChunkNodes from the 
       strategy's results and attaches them to the correct parent nodes in the 
       document tree, maintaining structural integrity.
    """
    
    def __init__(self, strategy: StrategyProtocol, alias: Optional[str] = None) -> None:
        """
        Initializes the pipeline step.

        Args:
            strategy: The processing strategy instance to be executed.
            alias: A unique identifier for this step. If not provided, it defaults 
                   to the strategy's class name. This alias is used for tagging 
                   results and configuring annotators.
        """
        self.strategy = strategy
        self.alias = alias or strategy.__class__.__name__
        self.annotators: Dict[str, StrategyProtocol] = {}

    def execute(
        self, 
        input_nodes: List[Union[DocumentRoot, ChunkNode]], 
        context: ExecutionContext
    ) -> List[ChunkNode]:
        """
        Executes the strategy logic across the provided frontier of input nodes.

        This method acts as the main entry point for data processing within this step. 
        It aggregates data from the input nodes, sets up the execution context with 
        local annotators, and dispatches the data to the underlying strategy using 
        the appropriate execution mode.

        Args:
            input_nodes: The list of nodes from the previous step (or the document root) 
                         that contain the input data for this strategy.
            context: The shared execution environment containing runtime dependencies 
                     and global state.

        Returns:
            A list of newly created ChunkNodes resulting from the execution. These 
            nodes represent the next frontier in the processing graph.
        """
        new_nodes = []
        
        inputs_with_parents = []
        
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
        Handles the execution of strategies that support batch processing.

        In this mode, all input items are aggregated into a single list and passed 
        to the strategy. This is particularly efficient for strategies that can 
        parallelize work or optimize bulk operations (like vectorized embeddings 
        or LLM calls).

        Critically, this method relies on the strategy returning payloads that 
        correctly populate their 'children' field. This field is used to map 
        the result back to its specific parent node using the lineage map.
        """
        all_inputs = [item for item, _ in inputs_with_parents]
        lineage_map = {
            item.id: pid 
            for item, pid in inputs_with_parents 
            if isinstance(item, ChunkPayload)
        }
        raw_results = self.strategy.execute(all_inputs, context)
        
        if not isinstance(raw_results, list):
            raw_results = [raw_results]
            
        new_nodes = []
        for result_payload in raw_results:
            parent_id = self._resolve_parent_id(result_payload, lineage_map)
            
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
        Handles the execution of strategies that process items sequentially.

        In this mode, the pipeline iterates through each input item, calls the 
        strategy, and immediately creates the resulting nodes. Parent lineage 
        is explicitly tracked via the loop variable, making this method robust 
        for simple strategies that do not maintain internal lineage history.
        """
        new_nodes = []
        for item, p_id in inputs_with_parents:
            raw_result = self.strategy.execute(item, context)
            
            if isinstance(raw_result, list):
                results_list = raw_result
            else:
                results_list = [raw_result]
            
            for res in results_list:
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
        """
        Attaches a newly created child node to its corresponding parent node within 
        the origin nodes list. This builds the actual graph structure in memory.
        """
        if parent_id is None:
            return
        for origin in origin_nodes:
            if origin.id == parent_id:
                origin.branches.append(child_node)
                break

    def _supports_batch_input(self, strategy: StrategyProtocol) -> bool:
        """
        Introspects the strategy's 'execute' method to determine if it accepts a list input.

        This method analyzes the type hints of the strategy's arguments (excluding 
        'self' and 'context'). It robustly handles complex type definitions, including 
        unions and generics, to determine if 'List' is a valid input type.
        """
        try:
            sig = inspect.signature(strategy.execute)
            type_hints = get_type_hints(strategy.execute)
            
            for name, param in sig.parameters.items():
                if name == 'self' or name == 'context':
                    continue
                
                hint = type_hints.get(name, param.annotation)
                return self._type_allows_list(hint)
                
            return False 
            
        except Exception:
            return False

    def _type_allows_list(self, type_hint: Any) -> bool:
        """
        Recursively checks if a given type hint is or contains a List type.
        Supports standard List, typing.List, and Union types.
        """
        if type_hint is Any:
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
        lineage_map: Dict[uuid.UUID, uuid.UUID]
    ) -> Optional[uuid.UUID]:
        """
        Resolves the parent node ID for a given payload by examining its lineage.

        This method looks at the payload's 'children' (which represent the source 
        data) and maps the first child's ID back to the parent node ID using 
        the lineage map created before batch execution.
        """
        if payload.children:
            primary_source = payload.children[0]
            if primary_source.id in lineage_map:
                return lineage_map[primary_source.id]
        return None

    def _get_node_class(self, payloads: List[ChunkPayload]) -> Type[ChunkNode]:
        """
        Dynamically determines the specialized ChunkNode subclass to use for a result.

        This allows strategies to define custom node types (via a '__node_class__' 
        attribute on the payload) if they require storage logic beyond the 
        standard ChunkNode.
        """
        if payloads and hasattr(payloads[0], "__node_class__"):
            return getattr(payloads[0], "__node_class__")
        return ChunkNode

    def _create_node(
        self, 
        payloads: List[ChunkPayload], 
        parent_id: Optional[uuid.UUID] = None
    ) -> ChunkNode:
        """
        Instantiates a new node containing the processed payloads.

        This method handles the transfer of data from the temporary processing 
        payloads into the persistent graph node structure, ensuring that metadata 
        fields match the definition of the target node class.
        """
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
        Overloads the '>>' operator to define a sequential topology.
        Returns a list representing the sequence [current_step, next_step_or_list].
        """
        return [self, next_val]

    def __or__(self, annotator_step: 'Step') -> 'Step':
        """
        Overloads the '|' operator to attach an annotator strategy to this step.
        The annotated strategy will run within the context of this step's execution.

        Raises:
            PipelineConfigurationError: If an annotator with the same alias already exists.
        """
        if annotator_step.alias in self.annotators:
            raise PipelineConfigurationError(f"Duplicate annotator alias '{annotator_step.alias}'")
        self.annotators[annotator_step.alias] = annotator_step.strategy
        return self


class PumpkingPipeline:
    """
    Orchestrates the end-to-end execution of document processing tasks.

    This class interprets the topology defined by Step objects (including sequential 
    chains and parallel branches) and manages the lifecycle of the execution frontier. 
    It maintains the flow of data from the document root through all configured steps 
    until completion.
    """

    def __init__(self, structure: Union[Step, List[Any]]) -> None:
        """
        Initializes the pipeline with a defined structure.

        Args:
            structure: The topology of the pipeline, which can be a single Step 
                       or a list of Steps (and nested lists for branching).
        """
        self.steps = self._initialize_topology(structure)

    def _initialize_topology(self, structure: Any) -> List[Any]:
        """
        Normalizes the input structure into a flat list of executable steps.
        """
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
        Executes the pipeline on the given input data.

        This method initializes the execution context and processes the document 
        layer by layer (frontier execution). At each stage, it identifies the 
        current active nodes and passes them to the next step(s) in the topology.

        Args:
            input_data: The raw document string or a pre-initialized DocumentRoot.
            filename: Optional filename metadata to attach if creating a new root.

        Returns:
            The fully processed DocumentRoot containing the complete graph of 
            ChunkNodes and results.
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
                for sub_step in step_item:
                    next_frontier.extend(sub_step.execute(current_frontier, current_context))
            else:
                next_frontier.extend(step_item.execute(current_frontier, current_context))
            
            current_frontier = next_frontier
            
        return root

    def __rshift__(self, next_val: Union[Step, List[Any]]) -> 'PumpkingPipeline':
        """
        Allows appending steps to an already instantiated pipeline object using 
        the '>>' operator.

        Args:
            next_val: A Step or list of Steps to append to the execution sequence.

        Returns:
            The pipeline instance itself, allowing for fluent chaining.

        Raises:
            PipelineConfigurationError: If the appended value has an invalid type.
        """
        if isinstance(next_val, Step):
            self.steps.append(next_val)
        elif isinstance(next_val, list):
            self.steps.append(next_val)
        else:
            raise PipelineConfigurationError(f"Invalid type for pipeline chaining: {type(next_val)}")
        return self


def annotate(strategy: Any, alias: str) -> Step:
    """
    Factory function to create a Step configured specifically for annotation.

    This helper simplifies the syntax for defining annotators that are attached 
    to other steps using the '|' operator.

    Args:
        strategy: The strategy logic to be used for annotation.
        alias: The name under which the annotation results will be stored.

    Returns:
        A new Step instance wrapping the strategy.
    """
    return Step(strategy, alias=alias)