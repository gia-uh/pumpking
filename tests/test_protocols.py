from typing import List, Union, get_type_hints, Any
from pumpking.models import ChunkPayload
from pumpking.protocols import ExecutionContext
from pumpking.strategies.base import BaseStrategy


class MockStrictStrategy(BaseStrategy):
    """
    A mock strategy that enforces strict string input.
    Used to verify that the introspection logic correctly identifies
    single-type constraints in the execute method signature.
    """

    def execute(self, data: str, context: ExecutionContext) -> List[ChunkPayload]:
        return self._apply_annotators_to_list([data], context)


class MockFlexibleStrategy(BaseStrategy):
    """
    A mock strategy accepting a Union of types.
    Used to verify that the system correctly handles polymorphic input definitions,
    which are common in strategies that support chaining (receiving either raw text
    or previous payloads).
    """

    def execute(self, data: Union[str, ChunkPayload], context: ExecutionContext) -> List[ChunkPayload]:
        content = data.content if isinstance(data, ChunkPayload) else str(data)
        return self._apply_annotators_to_list([content], context)


class MockAnnotator(BaseStrategy):
    """
    A mock strategy acting as an annotator.
    It returns a simple string to verify that metadata injection works
    through the base class helper methods.
    """

    def execute(self, data: str, context: ExecutionContext) -> str:
        return f"Annotated: {data}"


def test_strategy_signature_introspection():
    """
    Verifies that the runtime type introspection correctly resolves the
    type hints defined in the 'execute' method of a strategy.

    This test asserts that the 'data' argument's type annotation is
    accurately retrieved using the standard typing utilities. This is
    critical because the Pipeline relies on this mechanism to determine
    compatibility and execution modes (Batch vs. Iterative).
    """
    hints = get_type_hints(MockStrictStrategy.execute)
    input_type = hints.get("data")

    assert input_type is str


def test_polymorphic_signature_introspection():
    """
    Verifies that complex type hints (Unions) are correctly preserved
    during introspection.

    This ensures that the validation logic can handle strategies that
    accept multiple input formats (e.g., Union[str, ChunkPayload]),
    confirming that the architecture supports flexible component design.
    """
    hints = get_type_hints(MockFlexibleStrategy.execute)
    input_type = hints.get("data")

    assert input_type == Union[str, ChunkPayload]


def test_base_helper_execution_flow_with_context():
    """
    Verifies the functionality of the base class helper methods for
    payload construction and annotation injection.

    This test sets up an execution context with an active annotator
    and executes a strategy. It asserts that:
    1. The strategy produces the correct number of payloads.
    2. The payloads are instances of ChunkPayload.
    3. The annotator was successfully executed and its result injected
       into the payload's annotations dictionary.
    """
    annotator = MockAnnotator()
    context = ExecutionContext(annotators={"meta": annotator})
    strategy = MockStrictStrategy()

    results = strategy.execute("Test Content", context)

    assert len(results) == 1
    payload = results[0]

    assert isinstance(payload, ChunkPayload)
    assert payload.content == "Test Content"
    assert "meta" in payload.annotations
    assert payload.annotations["meta"] == "Annotated: Test Content"