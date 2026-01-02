import pytest
from typing import Any, List
from pumpking.pipeline import Step, PumpkingPipeline, annotate
from pumpking.protocols import ExecutionContext
from pumpking.strategies.base import BaseStrategy
from pumpking.validation import validate
from pumpking.exceptions import PipelineConfigurationError
from pumpking.models import ChunkPayload

# --- Mocks and Data ---


class StringProducer(BaseStrategy):
    """Produces str."""
    
    def execute(self, data: Any, context: ExecutionContext) -> str:
        return "data"


class StringConsumer(BaseStrategy):
    """Consumes str."""

    def execute(self, data: str, context: ExecutionContext) -> str:
        return data


class ListConsumer(BaseStrategy):
    """Consumes list."""

    def execute(self, data: list, context: ExecutionContext) -> str:
        return "merged"


class IntConsumer(BaseStrategy):
    """Consumes int (Incompatible with str/Payload)."""

    def execute(self, data: int, context: ExecutionContext) -> int:
        return 1


# --- Tests ---


def test_valid_linear_pipeline():
    """Test A(str) -> B(str) passes validation."""
    pipeline = Step(StringProducer()) >> Step(StringConsumer())
    try:
        validate(pipeline)
    except PipelineConfigurationError:
        pytest.fail("Valid pipeline raised PipelineConfigurationError")


def test_invalid_linear_pipeline():
    """Test A(str) -> B(int) raises error."""
    pipeline = Step(StringProducer()) >> Step(IntConsumer())

    with pytest.raises(PipelineConfigurationError) as excinfo:
        validate(pipeline)

    assert "Type Mismatch" in str(excinfo.value)


def test_valid_parallel_split():
    """Test A(str) -> [B(str), C(str)] passes."""
    pipeline = Step(StringProducer()) >> [
        Step(StringConsumer(), alias="consumer_v1"),
        Step(StringConsumer(), alias="consumer_v2"),
    ]
    validate(pipeline)


def test_invalid_parallel_split():
    """Test A(str) -> [B(str), C(int)] raises error on branch C."""
    pipeline = Step(StringProducer()) >> [
        Step(StringConsumer(), alias="valid_branch"),
        Step(IntConsumer(), alias="invalid_branch"),
    ]

    with pytest.raises(PipelineConfigurationError) as excinfo:
        validate(pipeline)

    assert "Branch 1" in str(excinfo.value)


def test_valid_merge_explicit_list():
    """
    Test [A, B] -> C(list).
    Parallel blocks output a 'list' of results, and C accepts list explicitly.
    """
    parallel_block = [
        Step(StringProducer(), alias="producer_A"),
        Step(StringProducer(), alias="producer_B"),
    ]
    pipeline = PumpkingPipeline(Step(StringProducer()) >> parallel_block) >> Step(
        ListConsumer()
    )
    validate(pipeline)


def test_merge_with_implicit_flattening():
    """
    Test [A, B] -> C(str).
    
    PREVIOUSLY INVALID: C expected str but got list.
    NOW VALID: The pipeline supports Auto-FlatMap. Validation should PASS.
    """
    parallel_block = [Step(StringProducer(), alias="single_parallel")]
    pipeline = PumpkingPipeline(Step(StringProducer()) >> parallel_block)