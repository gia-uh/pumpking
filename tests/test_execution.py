from typing import Any, List
from pumpking.pipeline import Step, PumpkingPipeline, annotate
from pumpking.strategies.base import BaseStrategy
from pumpking.models import ChunkNode, ChunkPayload, DocumentRoot
from pumpking.protocols import ExecutionContext

class PassthroughStrategy(BaseStrategy):
    SUPPORTED_INPUTS = [str]
    PRODUCED_OUTPUT = str
    def execute(self, data, context):
        return data

class SplitStrategy(BaseStrategy):
    SUPPORTED_INPUTS = [str]
    PRODUCED_OUTPUT = list
    def execute(self, data, context):
        return data.split(",")

class SentimentMock(BaseStrategy):
    SUPPORTED_INPUTS = [str]
    PRODUCED_OUTPUT = float
    def execute(self, data, context):
        return 0.9

class PayloadCreatorStrategy(BaseStrategy):
    SUPPORTED_INPUTS = [str]
    PRODUCED_OUTPUT = ChunkPayload
    
    def execute(self, data, context):
        return self._apply_annotators_to_payload(data, context)

class SpyStrategy(BaseStrategy):
    SUPPORTED_INPUTS = [str]
    PRODUCED_OUTPUT = str
    
    def __init__(self):
        self.received_data = []

    def execute(self, data, context):
        self.received_data.append(data)
        return data

def test_pipeline_creates_root_node():
    pipeline = PumpkingPipeline([])
    root = pipeline.run("start_content")
    
    assert isinstance(root, DocumentRoot)
    assert len(root.children) == 1
    assert root.children[0].content == "start_content"
    assert root.document == "start_content"

def test_pipeline_flow_execution():
    spy_step_1 = SpyStrategy()
    spy_step_2 = SpyStrategy()
    
    pipeline = Step(spy_step_1) >> Step(spy_step_2)
    pipeline.run("initial")
    
    assert spy_step_1.received_data == ["initial"]
    assert spy_step_2.received_data == ["initial"]

def test_pipeline_branching_logic():
    spy_consumer = SpyStrategy()
    
    pipeline = Step(SplitStrategy()) >> Step(spy_consumer)
    
    pipeline.run("A,B,C")
    
    assert len(spy_consumer.received_data) == 3
    assert "A" in spy_consumer.received_data
    assert "B" in spy_consumer.received_data
    assert "C" in spy_consumer.received_data

def test_annotations_via_payload_transfer():
    step = Step(PayloadCreatorStrategy()) | annotate(SentimentMock(), alias="score")
    
    root = step.run("test_data")
    
    assert root.document == "test_data"
    assert root.children[0].children[0].annotations['score'] == 0.9