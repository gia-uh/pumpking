import pytest
from typing import Any, List
from pumpking.pipeline import Step
from pumpking.protocols import ExecutionContext
from pumpking.strategies.base import BaseStrategy
from pumpking.validation import validate
from pumpking.exceptions import PipelineConfigurationError

# --- Mocks ---

class StringProducer(BaseStrategy):
    """Produces str."""
    SUPPORTED_INPUTS = [str]
    PRODUCED_OUTPUT = str
    def execute(self, data, context): return "data"

class StringConsumer(BaseStrategy):
    """Consumes str."""
    SUPPORTED_INPUTS = [str]
    PRODUCED_OUTPUT = str
    def execute(self, data, context): return data

class ListConsumer(BaseStrategy):
    """Consumes list."""
    SUPPORTED_INPUTS = [list]
    PRODUCED_OUTPUT = str
    def execute(self, data, context): return "merged"

class IntConsumer(BaseStrategy):
    """Consumes int (Incompatible with str)."""
    SUPPORTED_INPUTS = [int]
    PRODUCED_OUTPUT = int
    def execute(self, data, context): return 1

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
    assert "produces '<class 'str'>'" in str(excinfo.value)

def test_valid_parallel_split():
    """Test A(str) -> [B(str), C(str)] passes."""
    pipeline = Step(StringProducer()) >> [Step(StringConsumer()), Step(StringConsumer())]
    validate(pipeline)

def test_invalid_parallel_split():
    """Test A(str) -> [B(str), C(int)] raises error on branch C."""
    pipeline = Step(StringProducer()) >> [Step(StringConsumer()), Step(IntConsumer())]
    
    with pytest.raises(PipelineConfigurationError) as excinfo:
        validate(pipeline)
    
    assert "Branch 1" in str(excinfo.value)

def test_valid_merge():
    """
    Test [A, B] -> C(list).
    Parallel blocks output a 'list' of results.
    """
    parallel_block = [Step(StringProducer()), Step(StringProducer())]
    pipeline = Step(StringProducer()) >> parallel_block >> Step(ListConsumer())
    validate(pipeline)

def test_invalid_merge():
    """
    Test [A, B] -> C(str).
    Parallel block outputs 'list', but C expects 'str'.
    """
    parallel_block = [Step(StringProducer())]
    pipeline = Step(StringProducer()) >> parallel_block >> Step(StringConsumer())
    
    with pytest.raises(PipelineConfigurationError) as excinfo:
        validate(pipeline)
        
    assert "produces '<class 'list'>'" in str(excinfo.value)

def test_annotator_compatibility():
    """Test that annotators must accept str."""
    # Valid annotator (accepts str)
    s1 = Step(StringProducer())
    s1.annotators["valid"] = StringConsumer()
    validate(s1 >> Step(StringConsumer()))

    # Invalid annotator (accepts int only)
    s2 = Step(StringProducer())
    s2.annotators["invalid"] = IntConsumer()
    
    with pytest.raises(PipelineConfigurationError) as excinfo:
        validate(s2 >> Step(StringConsumer()))
        
    assert "Annotator 'invalid'" in str(excinfo.value)