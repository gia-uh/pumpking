from typing import List, Union, Type, Any
from pumpking.pipeline import Step, PumpkingPipeline
from pumpking.exceptions import PipelineConfigurationError

def validate(pipeline: Union[Step, List[Any], PumpkingPipeline]) -> None:
    """
    Performs static analysis on the pipeline structure.
    
    Checks for:
    1. Type compatibility between sequential steps.
    2. Input type compatibility for parallel branches.
    3. Alias uniqueness among sibling steps in parallel blocks.
    4. Annotator compatibility (must accept string input).
    
    Args:
        pipeline: The definition of the pipeline flow.
        
    Raises:
        PipelineConfigurationError: If any validation rule is violated.
    """
    steps = _normalize_pipeline(pipeline)
    
    current_output_type = str
    
    for index, item in enumerate(steps):
        if isinstance(item, Step):
            _validate_step_compatibility(item, current_output_type)
            _validate_annotators(item)
            current_output_type = item.strategy.PRODUCED_OUTPUT
            
        elif isinstance(item, list):
            _validate_parallel_block(item, current_output_type)
            current_output_type = list
        else:
            raise PipelineConfigurationError(f"Invalid item at index {index}: {item}")

def _normalize_pipeline(pipeline: Union[Step, List[Any], PumpkingPipeline]) -> List[Any]:
    """Converts the input into a standard list of steps/blocks."""
    if isinstance(pipeline, PumpkingPipeline):
        return pipeline.steps
    if isinstance(pipeline, Step):
        return [pipeline]
    if isinstance(pipeline, list):
        return pipeline
    raise PipelineConfigurationError(f"Unknown pipeline type: {type(pipeline)}")

def _validate_step_compatibility(step: Step, input_type: Type) -> None:
    """
    Verifies that the step's strategy supports the type produced by the previous step.
    """
    supported = step.strategy.SUPPORTED_INPUTS
    
    if Any in supported:
        return

    if input_type not in supported:
        raise PipelineConfigurationError(
            f"Type Mismatch: Step '{step.alias}' expects {supported} "
            f"but receives '{input_type}'."
        )

def _validate_annotators(step: Step) -> None:
    """
    Verifies that all annotators attached to the step support string input,
    as annotators operate on the raw content payload.
    """
    for alias, strategy in step.annotators.items():
        supported = strategy.SUPPORTED_INPUTS
        if str not in supported and Any not in supported:
            raise PipelineConfigurationError(
                f"Annotator '{alias}' expects {supported} but receives 'str'."
            )

def _validate_parallel_block(branches: List[Step], input_type: Type) -> None:
    """
    Validates a parallel block for alias collisions and type safety.
    """
    aliases = set()
    
    for idx, step in enumerate(branches):
        if not isinstance(step, Step):
            raise PipelineConfigurationError(f"Parallel block contains non-Step item at index {idx}")
            
        if step.alias in aliases:
            raise PipelineConfigurationError(f"Duplicate step alias '{step.alias}'")
        aliases.add(step.alias)
        
        try:
            _validate_step_compatibility(step, input_type)
            _validate_annotators(step)
        except PipelineConfigurationError as e:
            raise PipelineConfigurationError(f"Branch {idx}: {str(e)}")