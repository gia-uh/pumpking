from typing import Any, List
from pumpking.models import ChunkPayload
from pumpking.protocols import ExecutionContext, StrategyProtocol
from pumpking.strategies.base import BaseStrategy

class MockAnnotator(BaseStrategy):
    """Mock strategy producing string output."""

    PRODUCED_OUTPUT = str

    def execute(self, data: Any, context: ExecutionContext) -> str:
        return f"Annotated: {data}"


class MockChunkingStrategy(BaseStrategy):
    """Mock strategy using default types (List[ChunkPayload])."""

    def execute(self, data: Any, context: ExecutionContext) -> List[ChunkPayload]:
        return self._apply_annotators_to_list([str(data)], context)


def test_protocol_uses_real_types():
    """Verify that attributes are real types."""
    assert str in MockChunkingStrategy.SUPPORTED_INPUTS
    assert List[str] in MockChunkingStrategy.SUPPORTED_INPUTS
    assert MockChunkingStrategy.PRODUCED_OUTPUT == List[ChunkPayload]


def test_type_compatibility_check():
    """Simulate the Static Validator logic with types."""
    producer_output = str
    consumer_inputs = MockChunkingStrategy.SUPPORTED_INPUTS
    is_valid = producer_output in consumer_inputs
    assert is_valid is True


def test_helper_execution_flow():
    """Verify helpers work with the new structure."""
    annotator = MockAnnotator()
    context = ExecutionContext(annotators={"meta": annotator})
    host = MockChunkingStrategy()

    result = host.execute("Test", context)

    assert len(result) == 1
    assert isinstance(result[0], ChunkPayload)
    assert result[0].annotations["meta"] == "Annotated: Test"