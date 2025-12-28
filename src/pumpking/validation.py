from typing import Any, List, Union, Set

from pumpking.pipeline import PumpkingPipeline, Step
from pumpking.exceptions import PipelineConfigurationError


def validate(pipeline: PumpkingPipeline) -> None:
    """
    Validates the type compatibility and structural integrity of a PumpkingPipeline.

    Checks:
    1. Internal Integrity: Validates annotators and unique aliases in parallel blocks.
    2. Sequence Compatibility: Ensures output types match input types between steps.

    Args:
        pipeline: The pipeline instance to validate.

    Raises:
        PipelineConfigurationError: If a configuration error is detected.
    """
    steps = pipeline.steps
    if not steps:
        return

    # 1. Validate Internal Integrity
    for step_block in steps:
        _validate_block_annotators(step_block)
        _validate_unique_aliases(step_block)

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
    """
    for alias, annotator in step.annotators.items():
        if str not in annotator.SUPPORTED_INPUTS:
            raise PipelineConfigurationError(
                f"Annotator '{alias}' (on {step.alias}) requires input types "
                f"{annotator.SUPPORTED_INPUTS}, but annotators receive 'str'."
            )


def _validate_unique_aliases(block: Union[Step, List[Step]]) -> None:
    """
    Ensures that all steps within a parallel block have unique aliases.
    Duplicate aliases in parallel execution would cause data overwrites during merging.
    """
    if not isinstance(block, list):
        return

    seen_aliases: Set[str] = set()
    for step in block:
        if step.alias in seen_aliases:
            raise PipelineConfigurationError(
                f"Duplicate step alias '{step.alias}' found in parallel block. "
                "Sibling steps must have unique aliases to merge results correctly."
            )
        seen_aliases.add(step.alias)


def _get_block_output_type(block: Union[Step, List[Step]]) -> Any:
    """
    Determines the output type of a block.
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