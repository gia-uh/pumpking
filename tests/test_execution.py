from typing import Any, List
from pumpking.pipeline import Step, PumpkingPipeline, annotate
from pumpking.strategies.base import BaseStrategy
from pumpking.models import ChunkNode

class PassthroughStrategy(BaseStrategy):
    """Returns input as is."""
    SUPPORTED_INPUTS = [str]
    PRODUCED_OUTPUT = str
    def execute(self, data, context):
        return data

class SplitStrategy(BaseStrategy):
    """Splits string by comma."""
    SUPPORTED_INPUTS = [str]
    PRODUCED_OUTPUT = list
    def execute(self, data, context):
        return data.split(",")

class SentimentMock(BaseStrategy):
    """Returns a fake sentiment score."""
    SUPPORTED_INPUTS = [str]
    PRODUCED_OUTPUT = float
    def execute(self, data, context):
        return 0.9

class SpyStrategy(BaseStrategy):
    """Records executions to validation."""
    SUPPORTED_INPUTS = [str]
    PRODUCED_OUTPUT = str
    
    def __init__(self):
        self.received_data = []

    def execute(self, data, context):
        self.received_data.append(data)
        return data

def test_pipeline_creates_root_node():
    """Verify that run() returns a root node with the initial input."""
    pipeline = PumpkingPipeline([])
    root = pipeline.run("start_content")
    
    assert isinstance(root, ChunkNode)
    assert root.content == "start_content"
    assert root.parent_id is None

def test_pipeline_flow_execution():
    """
    Verify that strategies are executed in order.
    """
    spy_step_1 = SpyStrategy()
    spy_step_2 = SpyStrategy()
    
    pipeline = Step(spy_step_1) >> Step(spy_step_2)
    pipeline.run("initial")
    
    assert spy_step_1.received_data == ["initial"]
    assert spy_step_2.received_data == ["initial"]

def test_pipeline_branching_logic():
    """
    Verify that a split strategy creates multiple children nodes 
    processed by the next step.
    """
    spy_consumer = SpyStrategy()
    
    pipeline = Step(SplitStrategy()) >> Step(spy_consumer)
    
    pipeline.run("A,B,C")
    
    assert len(spy_consumer.received_data) == 3
    assert "A" in spy_consumer.received_data
    assert "B" in spy_consumer.received_data
    assert "C" in spy_consumer.received_data

def test_annotators_are_applied_by_pipeline_on_single_step():
    """
    Verify that we can run a single step with annotations directly
    without manually wrapping it in a PumpkingPipeline.
    """
    class NodeInspector(BaseStrategy):
        SUPPORTED_INPUTS = [str]
        PRODUCED_OUTPUT = str
        def execute(self, data, context):
            return data

    inspector = NodeInspector()
    
    pipeline_step = Step(PassthroughStrategy()) | annotate(SentimentMock(), alias="score")
    
    root = pipeline_step.run("test_data")
    
    assert root.content == "test_data"