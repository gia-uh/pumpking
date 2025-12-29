from pumpking.pipeline import PumpkingPipeline, Step, annotate
from pumpking.models import ChunkNode, DocumentRoot, ChunkPayload
from pumpking.strategies.base import BaseStrategy
from pumpking.protocols import ExecutionContext

class SpyStrategy(BaseStrategy):
    SUPPORTED_INPUTS = [str]
    PRODUCED_OUTPUT = str

    def execute(self, data: str, context: ExecutionContext) -> str:
        return data

class SplitStrategy(BaseStrategy):
    SUPPORTED_INPUTS = [str]
    PRODUCED_OUTPUT = list

    def execute(self, data: str, context: ExecutionContext) -> list:
        return data.split(",")

class PayloadCreatorStrategy(BaseStrategy):
    def execute(self, data: str, context: ExecutionContext) -> ChunkPayload:
        return self._apply_annotators_to_payload(data, context)

class SentimentMock(BaseStrategy):
    def execute(self, data: str, context: ExecutionContext) -> float:
        return 0.9

def test_pipeline_creates_root_node():
    pipeline = PumpkingPipeline([])
    root = pipeline.run("start_content")
    
    assert isinstance(root, DocumentRoot)
    assert root.document == "start_content"
    assert len(root.children) == 1
    assert root.children[0].content == "start_content"

def test_pipeline_flow_execution():
    spy_step_1 = SpyStrategy()
    spy_step_2 = SpyStrategy()
    
    pipeline = Step(spy_step_1) >> Step(spy_step_2)
    root = pipeline.run("initial")
    
    assert len(root.children) == 1
    first_level = root.children[0]
    assert len(first_level.children) == 1
    second_level = first_level.children[0]
    assert len(second_level.children) == 1

def test_pipeline_branching_logic():
    spy_consumer = SpyStrategy()
    
    pipeline = Step(SplitStrategy()) >> Step(spy_consumer)
    
    root = pipeline.run("A,B,C")
    
    base_node = root.children[0]
    assert len(base_node.children) == 3
    assert base_node.children[0].content == "A"
    assert base_node.children[1].content == "B"
    assert base_node.children[2].content == "C"

def test_annotations_via_payload_transfer():
    step = Step(PayloadCreatorStrategy()) | annotate(SentimentMock(), alias="score")
    
    root = step.run("test_data")
    
    assert root.document == "test_data"
    
    processed_node = root.children[0].children[0]
    assert processed_node.content == "test_data"
    assert processed_node.annotations['score'] == 0.9