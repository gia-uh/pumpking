from typing import Any, List, Union, Set

from pumpking.pipeline import PumpkingPipeline, Step
from pumpking.exceptions import PipelineConfigurationError
from pumpking.models import ChunkPayload


def validate(pipeline: PumpkingPipeline) -> None:
    steps = pipeline.steps
    if not steps:
        return

    for step_block in steps:
        _validate_block_annotators(step_block)
        _validate_unique_aliases(step_block)

    for i in range(len(steps) - 1):
        current_block = steps[i]
        next_block = steps[i+1]
        
        output_type = _get_block_output_type(current_block)
        _validate_connection(output_type, next_block, step_index=i)


def _validate_block_annotators(block: Union[Step, List[Step]]) -> None:
    if isinstance(block, list):
        for step in block:
            _check_annotator_compatibility(step)
    else:
        _check_annotator_compatibility(block)


def _check_annotator_compatibility(step: Step) -> None:
    for alias, annotator in step.annotators.items():
        if str not in annotator.SUPPORTED_INPUTS:
            raise PipelineConfigurationError(
                f"Annotator '{alias}' (on {step.alias}) requires input types "
                f"{annotator.SUPPORTED_INPUTS}, but annotators receive 'str'."
            )


def _validate_unique_aliases(block: Union[Step, List[Step]]) -> None:
    if not isinstance(block, list):
        return

    seen_aliases: Set[str] = set()
    for step in block:
        if step.alias in seen_aliases:
            raise PipelineConfigurationError(
                f"Duplicate step alias '{step.alias}' found in parallel block."
            )
        seen_aliases.add(step.alias)


def _get_block_output_type(block: Union[Step, List[Step]]) -> Any:
    if isinstance(block, list):
        return list
    return block.strategy.PRODUCED_OUTPUT


def _validate_connection(output_type: Any, next_block: Union[Step, List[Step]], step_index: int) -> None:
    def is_compatible(out_type: Any, supported_inputs: List[Any]) -> bool:
        if out_type in supported_inputs:
            return True
        if (out_type == ChunkPayload or out_type == List[ChunkPayload]) and str in supported_inputs:
            return True
        return False

    if isinstance(next_block, list):
        for branch_idx, step in enumerate(next_block):
            if not is_compatible(output_type, step.strategy.SUPPORTED_INPUTS):
                # RESTORED: Detailed error message
                raise PipelineConfigurationError(
                    f"Type Mismatch at Step {step_index} -> {step_index + 1} (Branch {branch_idx}): "
                    f"Previous step produces '{output_type}', but "
                    f"'{step.alias}' only supports {step.strategy.SUPPORTED_INPUTS}."
                )
    else:
        if not is_compatible(output_type, next_block.strategy.SUPPORTED_INPUTS):
            # RESTORED: Detailed error message
            raise PipelineConfigurationError(
                f"Type Mismatch at Step {step_index} -> {step_index + 1}: "
                f"Previous step produces '{output_type}', but "
                f"'{next_block.alias}' only supports {next_block.strategy.SUPPORTED_INPUTS}."
            )