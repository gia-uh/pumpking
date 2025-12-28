from typing import Any, List
from pumpking.pipeline import Step, PumpkingPipeline
from pumpking.protocols import ExecutionContext
from pumpking.strategies.base import BaseStrategy

class PassthroughStrategy(BaseStrategy):
    """Returns input as is."""
    SUPPORTED_INPUTS = [Any]
    PRODUCED_OUTPUT = Any
    def execute(self, data, context):
        return data

class UpperCaseStrategy(BaseStrategy):
    """Converts string to uppercase."""
    SUPPORTED_INPUTS = [str]
    PRODUCED_OUTPUT = str
    def execute(self, data, context):
        return data.upper()

class SplitStrategy(BaseStrategy):
    """Splits string into list."""
    SUPPORTED_INPUTS = [str]
    PRODUCED_OUTPUT = list
    def execute(self, data, context):
        return data.split(",")

class SumStrategy(BaseStrategy):
    """Sums a list of numbers."""
    SUPPORTED_INPUTS = [list]
    PRODUCED_OUTPUT = int
    def execute(self, data, context):
        return sum(data)

class BranchToString(BaseStrategy):
    """Converts any input to string for branch testing."""
    SUPPORTED_INPUTS = [Any]
    PRODUCED_OUTPUT = str
    def execute(self, data, context):
        return str(data)

def test_linear_execution_flow():
    """Verify data flows sequentially through steps."""
    pipeline = Step(PassthroughStrategy()) >> Step(UpperCaseStrategy())
    result = pipeline.run("hello")
    assert result == "HELLO"

def test_parallel_execution_flow():
    """Verify data is broadcast to parallel branches."""
    pipeline = Step(PassthroughStrategy()) >> [
        Step(BranchToString(), alias="v1"),
        Step(BranchToString(), alias="v2")
    ]
    result = pipeline.run("test")
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == "test"
    assert result[1] == "test"

def test_complex_execution_flow():
    """Verify Linear -> Parallel -> Linear flow."""
    pipeline = (
        Step(PassthroughStrategy()) 
        >> [Step(UpperCaseStrategy(), alias="upper"), Step(SplitStrategy(), alias="split")]
    )
    
    result = pipeline.run("a,b")
    
    assert isinstance(result, list)
    assert result[0] == "A,B"
    assert result[1] == ["a", "b"]

def test_state_persistence():
    """Verify objects are passed by reference/value correctly."""
    input_data = [1, 2, 3]
    pipeline = Step(SumStrategy()) >> Step(BranchToString())
    
    result = pipeline.run(input_data)
    assert result == "6"

def test_annotators_are_instantiated_in_context():
    """Verify that execution context receives the step's annotators."""
    class MockWithContextCheck(BaseStrategy):
        SUPPORTED_INPUTS = [str]
        PRODUCED_OUTPUT = str
        def execute(self, data, context):
            if "meta" in context.annotators:
                return "annotator_found"
            return "no_annotator"

    # CORRECTION: Wrap the single Step in a PumpkingPipeline to execute it
    step = Step(MockWithContextCheck())
    step.annotators["meta"] = PassthroughStrategy()
    
    pipeline = PumpkingPipeline([step])
    
    result = pipeline.run("start")
    assert result == "annotator_found"