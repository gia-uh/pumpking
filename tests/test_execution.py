from typing import List, Union
from pumpking.pipeline import PumpkingPipeline, Step, annotate
from pumpking.models import DocumentRoot, ChunkPayload
from pumpking.strategies.base import BaseStrategy
from pumpking.protocols import ExecutionContext


# --- Mocks and Data ---
class SpyStrategy(BaseStrategy):
    """
    A strategy designed to pass the input text through to the output.
    Updated to handle both str and ChunkPayload inputs for pipeline compatibility.
    """

    def execute(self, data: Union[str, ChunkPayload], context: ExecutionContext) -> ChunkPayload:
        content = data.content if isinstance(data, ChunkPayload) else str(data)
        return self._apply_annotators_to_payload(content, context)


class SplitStrategy(BaseStrategy):
    """
    A strategy that simulates a document splitting process.
    Updated to handle input normalization.
    """

    def execute(self, data: Union[str, ChunkPayload], context: ExecutionContext) -> List[ChunkPayload]:
        # Normalization
        content = data.content if isinstance(data, ChunkPayload) else str(data)
        
        parts = content.split(",")
        return self._apply_annotators_to_list(parts, context)


class PayloadCreatorStrategy(BaseStrategy):
    """
    A strategy that explicitly constructs a ChunkPayload.
    Updated to handle input normalization.
    """

    def execute(self, data: Union[str, ChunkPayload], context: ExecutionContext) -> ChunkPayload:
        # Normalization
        content = data.content if isinstance(data, ChunkPayload) else str(data)
        return self._apply_annotators_to_payload(content, context)


class SentimentMock(BaseStrategy):
    """
    A mock strategy functioning as an annotator.
    Annotators always receive the raw content string, so 'data: str' is correct here.
    """

    def execute(self, data: str, context: ExecutionContext) -> float:
        return 0.9


# --- Tests ---


def test_pipeline_initialization_and_root_branching():
    """
    Verifies the initialization of the DocumentRoot and the
    first-level branching of the execution graph.
    """
    pipeline = PumpkingPipeline(Step(SpyStrategy(), alias="FirstStep"))
    root = pipeline.run("base_content")

    assert isinstance(root, DocumentRoot)
    assert len(root.branches) == 1
    assert root.branches[0].strategy_label == "FirstStep"
    assert root.branches[0].results[0].content == "base_content"


def test_sequential_execution_depth():
    """
    Validates that the sequential chaining of steps results in a
    hierarchically deep tree structure.
    """
    s1 = Step(SpyStrategy(), alias="Level1")
    s2 = Step(SpyStrategy(), alias="Level2")

    pipeline = PumpkingPipeline(s1 >> s2)
    root = pipeline.run("sequential_data")

    first_level_node = root.branches[0]
    assert len(first_level_node.branches) == 1

    second_level_node = first_level_node.branches[0]
    assert second_level_node.strategy_label == "Level2"
    assert second_level_node.results[0].content == "sequential_data"


def test_multi_node_branching_logic():
    """
    Tests the pipeline's ability to handle strategies that produce
    multiple outputs from a single input.

    We verify that the SplitStrategy correctly produces sibling nodes attached
    to the root, and that the consumer processes each sibling independently.
    """
    splitter = Step(SplitStrategy(), alias="SplitStep")
    consumer = Step(SpyStrategy(), alias="Consumer")

    pipeline = PumpkingPipeline(splitter >> consumer)
    root = pipeline.run("A,B,C")

    assert len(root.branches) == 3

    labels = [node.results[0].content for node in root.branches]
    assert labels == ["A", "B", "C"]

    for split_node in root.branches:
        assert len(split_node.branches) == 1
        assert split_node.branches[0].strategy_label == "Consumer"
        assert split_node.branches[0].results[0].content == split_node.results[0].content


def test_step_level_annotation_isolation():
    """
    Verifies that annotations are applied correctly and scoped strictly
    to the Step where they are defined.
    """
    annotated_step = Step(PayloadCreatorStrategy()) | annotate(
        SentimentMock(), alias="score"
    )
    next_step = Step(SpyStrategy())

    pipeline = PumpkingPipeline(annotated_step >> next_step)
    root = pipeline.run("annotation_test")

    processed_node = root.branches[0]
    assert processed_node.results[0].annotations["score"] == 0.9

    child_node = processed_node.branches[0]
    assert "score" not in child_node.results[0].annotations