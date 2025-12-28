from typing import Any, List, Union, Type

from pumpking.pipeline import PumpkingPipeline, Step
from pumpking.exceptions import PipelineConfigurationError


def validate(pipeline: PumpkingPipeline) -> None:
    """
    Validates the type compatibility of a PumpkingPipeline before execution.

    Traverses the pipeline steps and ensures that the output produced by step N
    is compatible with the input supported by step N+1. It also validates that
    annotators attached to a step are compatible with string input (default for content).

    Args:
        pipeline: The pipeline instance to validate.

    Raises:
        PipelineConfigurationError: If a type mismatch is detected.
    """
    steps = pipeline.steps
    if not steps:
        return

    # 1. Validate Internal Integrity (Annotators)
    for step_block in steps:
        _validate_block_annotators(step_block)

    # 2. Validate Sequence Compatibility (Step N -> Step N+1)
    for i in range(len(steps) - 1):
        current_block = steps[i]
        next_block = steps[i+1]
        
        output_type = _get_block_output_type(current_block)
        _validate_connection(output_type, next_block, step_index=i)


def _validate_block_annotators(block: Union[Step, List[Step]]) -> None:
    """Checks if annotators within a step or list of steps are valid."""
    if isinstance(block, list):
        for step in block:
            _check_annotator_compatibility(step)
    else:
        _check_annotator_compatibility(block)


def _check_annotator_compatibility(step: Step) -> None:
    """
    Verifies that registered annotators support 'str' input.
    Pumpking annotators typically operate on the 'content' of a ChunkNode (str).
    """
    for alias, annotator in step.annotators.items():
        if str not in annotator.SUPPORTED_INPUTS:
            raise PipelineConfigurationError(
                f"Annotator '{alias}' (on {step.alias}) requires input types "
                f"{annotator.SUPPORTED_INPUTS}, but annotators receive 'str'."
            )


def _get_block_output_type(block: Union[Step, List[Step]]) -> Any:
    """
    Determines the output type of a block (Single Step or Parallel List).
    If it's a parallel list, the output is a standard Python list containing results.
    """
    if isinstance(block, list):
        return list
    return block.strategy.PRODUCED_OUTPUT


def _validate_connection(output_type: Any, next_block: Union[Step, List[Step]], step_index: int) -> None:
    """
    Checks if 'output_type' is acceptable for 'next_block'.
    """
    if isinstance(next_block, list):
        # Fan-out: Output must be valid for ALL branches in the parallel block
        for branch_idx, step in enumerate(next_block):
            if output_type not in step.strategy.SUPPORTED_INPUTS:
                raise PipelineConfigurationError(
                    f"Type Mismatch at Step {step_index} -> {step_index + 1} (Branch {branch_idx}): "
                    f"Previous step produces '{output_type}', but "
                    f"'{step.alias}' only supports {step.strategy.SUPPORTED_INPUTS}."
                )
    else:
        # Linear: Output must be valid for the next step
        if output_type not in next_block.strategy.SUPPORTED_INPUTS:
            raise PipelineConfigurationError(
                f"Type Mismatch at Step {step_index} -> {step_index + 1}: "
                f"Previous step produces '{output_type}', but "
                f"'{next_block.alias}' only supports {next_block.strategy.SUPPORTED_INPUTS}."
            )