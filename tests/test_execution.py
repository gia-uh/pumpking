from typing import List
from pumpking.pipeline import PumpkingPipeline, Step, annotate
from pumpking.models import DocumentRoot, ChunkPayload
from pumpking.strategies.base import BaseStrategy
from pumpking.protocols import ExecutionContext


# --- Mocks and Data ---
class SpyStrategy(BaseStrategy):
    """
    A strategy designed to pass the input text through to the output.

    In the Inversion of Control architecture, this strategy demonstrates
    compliance by utilizing the base class helper method to apply any
    annotators present in the execution context to the data. This ensures
    that even a simple pass-through strategy participates correctly in
    the metadata enrichment process.
    """

    SUPPORTED_INPUTS = [str]
    PRODUCED_OUTPUT = ChunkPayload

    def execute(self, data: str, context: ExecutionContext) -> ChunkPayload:
        return self._apply_annotators_to_payload(data, context)


class SplitStrategy(BaseStrategy):
    """
    A strategy that simulates a document splitting process.

    It transforms a single input string into a list of strings based on
    a delimiter. To adhere to the pipeline's annotation contract, it
    uses the list-specific helper method. This converts the raw string
    segments into fully annotated ChunkPayload objects before returning
    them, allowing the pipeline to generate multiple branch nodes with
    correct metadata.
    """

    SUPPORTED_INPUTS = [str]
    PRODUCED_OUTPUT = list

    def execute(self, data: str, context: ExecutionContext) -> List[ChunkPayload]:
        parts = data.split(",")
        return self._apply_annotators_to_list(parts, context)


class PayloadCreatorStrategy(BaseStrategy):
    """
    A strategy that explicitly constructs a ChunkPayload.

    This class verifies that the pipeline can handle pre-built payload
    objects. It explicitly invokes the annotation helper to ensure that
    the manually created payload still receives any metadata defined
    at the Step level, maintaining consistency between custom object
    creation and the pipeline's configuration.
    """

    def execute(self, text: str, context: ExecutionContext) -> ChunkPayload:
        return self._apply_annotators_to_payload(text, context)


class SentimentMock(BaseStrategy):
    """
    A mock strategy functioning as an annotator.

    It returns a fixed float value. When used within the pipeline's
    annotation system, this value is keyed into the payload's annotation
    dictionary. This mock allows tests to verify that specific logic
    units are being executed and their outputs are being correctly
    attached to the data stream.
    """

    def execute(self, data: str, context: ExecutionContext) -> float:
        return 0.9


# --- Tests ---


def test_pipeline_initialization_and_root_branching():
    """
    Verifies the initialization of the DocumentRoot and the
    first-level branching of the execution graph.

    The test validates that the pipeline correctly processes the initial
    input string into a DocumentRoot and that the first Step in the
    sequence successfully creates a child node attached to this root.
    It checks strictly for structural integrity and data fidelity
    at the start of the execution flow.
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

    This test ensures that the output of one step becomes the processing
    context for the next. By checking that the second step's node is
    a child of the first step's node (rather than a sibling), it confirms
    that the pipeline correctly manages the frontier of execution
    for sequential operations defined via the bitwise shift operator.
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

    By using a splitting strategy, this test confirms that the pipeline
    adapter correctly interprets a list of results (ChunkPayloads)
    and converts them into distinct sibling nodes in the graph.
    It verifies that the parent node's branch list contains exactly
    one node for each item produced by the strategy.
    """
    splitter = Step(SplitStrategy(), alias="SplitStep")
    consumer = Step(SpyStrategy(), alias="Consumer")

    pipeline = PumpkingPipeline(splitter >> consumer)
    root = pipeline.run("A,B,C")

    base_node = root.branches[0]
    assert len(base_node.branches) == 3

    labels = [node.results[0].content for node in base_node.branches]
    assert labels == ["A", "B", "C"]


def test_step_level_annotation_isolation():
    """
    Verifies that annotations are applied correctly and scoped strictly
    to the Step where they are defined.

    This test constructs a pipeline with two steps, attaching an
    annotator only to the first one. It asserts two critical conditions:
    1. The payload produced by the first step contains the expected annotation key and value.
    2. The payload produced by the second step does not contain this annotation.

    This confirms that the execution context is correctly managing the
    injection of annotators and that the strategies are respecting
    the inversion of control pattern by applying them.
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
