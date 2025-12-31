from typing import List
from pumpking.pipeline import Step, PumpkingPipeline, annotate
from pumpking.models import ChunkNode, ChunkPayload
from pumpking.protocols import ExecutionContext
from pumpking.strategies.base import BaseStrategy

# --- Mocks and Data ---


class MockSpecialNode(ChunkNode):
    """Generic specialized node for testing structural inheritance."""

    special_attr: str = "default"


class MockSpecialPayload(ChunkPayload):
    """Generic specialized payload providing the node class hint."""

    special_attr: str = "custom_value"
    __node_class__ = MockSpecialNode


class MockSplitter(BaseStrategy):
    """
    Returns multiple strings to verify branching.
    Uses helper to apply context annotations to list items.
    """

    def execute(self, text: str, context: ExecutionContext) -> List[ChunkPayload]:
        parts = ["part_a", "part_b"]
        return self._apply_annotators_to_list(parts, context)


class MockIdentityStrategy(BaseStrategy):
    """
    Returns the input as a single result.
    CRITICAL FIX: Explicitly applies annotators from context using helper.
    """

    def execute(self, text: str, context: ExecutionContext) -> ChunkPayload:
        return self._apply_annotators_to_payload(text, context)


class MockSpecStrategy(BaseStrategy):
    """Returns specialized payloads to verify the dynamic factory."""

    def execute(self, text: str, context: ExecutionContext) -> List[MockSpecialPayload]:
        return [MockSpecialPayload(content="data", content_raw="raw")]


class MockAnnotator(BaseStrategy):
    """Simple annotator strategy."""

    def execute(self, text: str, context: ExecutionContext) -> str:
        return "annotated"


# --- Tests ---


def test_pipeline_parallel_branching_logic():
    """
    Ensures that parallel steps ([Step, Step]) branch from the same parent.
    Verified by counting sibling branches in the graph.
    """
    pipeline = PumpkingPipeline(
        Step(MockSplitter(), alias="Parent")
        >> [Step(MockSplitter(), alias="Path1"), Step(MockSplitter(), alias="Path2")]
    )
    root = pipeline.run("Input")

    result = root
    parent_node = result.branches[0]

    assert len(parent_node.branches) == 4


def test_dynamic_node_specialization_factory():
    """
    Verifies that the factory correctly identifies a specialized node
    via class hints and maps attributes dynamically.
    """
    pipeline = PumpkingPipeline(Step(MockSpecStrategy()))
    root = pipeline.run("Special Data")

    node = root.branches[0]

    assert isinstance(node, MockSpecialNode)
    assert node.special_attr == "custom_value"


def test_pipeline_annotation_isolation():
    """
    Ensures annotators are local to the Step and do not diffuse to the next step.
    This test previously failed because the Mock ignored the context.
    """
    s1 = Step(MockIdentityStrategy(), alias="S1") | annotate(
        MockAnnotator(), alias="note"
    )
    s2 = Step(MockIdentityStrategy(), alias="S2")

    pipeline = PumpkingPipeline(s1 >> s2)
    root = pipeline.run("Check Annotations")

    node_s1 = root.branches[0]
    node_s2 = node_s1.branches[0]

    assert "note" in node_s1.results[0].annotations
    assert node_s1.results[0].annotations["note"] == "annotated"

    assert "note" not in node_s2.results[0].annotations


def test_pipeline_sequential_depth():
    """
    Verifies that the >> operator creates a deep tree.
    S1 should be a branch of Root, and S2 should be a branch of S1.
    """
    s1 = Step(MockIdentityStrategy(), alias="S1")
    s2 = Step(MockIdentityStrategy(), alias="S2")

    pipeline = PumpkingPipeline(s1 >> s2)
    root = pipeline.run("Input")

    assert len(root.branches) == 1
    node_s1 = root.branches[0]
    assert node_s1.strategy_label == "S1"

    assert len(node_s1.branches) == 1
    node_s2 = node_s1.branches[0]
    assert node_s2.strategy_label == "S2"
