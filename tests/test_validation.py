import pytest
from typing import Any, List
from pumpking.pipeline import Step, annotate
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
    # CORRECTION: Added explicit aliases to avoid naming collision error
    pipeline = Step(StringProducer()) >> [
        Step(StringConsumer(), alias="consumer_v1"), 
        Step(StringConsumer(), alias="consumer_v2")
    ]
    validate(pipeline)

def test_invalid_parallel_split():
    """Test A(str) -> [B(str), C(int)] raises error on branch C."""
    # CORRECTION: Added explicit aliases
    pipeline = Step(StringProducer()) >> [
        Step(StringConsumer(), alias="valid_branch"), 
        Step(IntConsumer(), alias="invalid_branch")
    ]
    
    with pytest.raises(PipelineConfigurationError) as excinfo:
        validate(pipeline)
    
    # Error should point to the invalid branch index or definition
    assert "Branch 1" in str(excinfo.value)

def test_valid_merge():
    """
    Test [A, B] -> C(list).
    Parallel blocks output a 'list' of results.
    """
    # CORRECTION: Added explicit aliases
    parallel_block = [
        Step(StringProducer(), alias="producer_A"), 
        Step(StringProducer(), alias="producer_B")
    ]
    pipeline = Step(StringProducer()) >> parallel_block >> Step(ListConsumer())
    validate(pipeline)

def test_invalid_merge():
    """
    Test [A, B] -> C(str).
    Parallel block outputs 'list', but C expects 'str'.
    """
    # CORRECTION: Added alias (though strictly only needed if >1 item)
    parallel_block = [Step(StringProducer(), alias="single_parallel")]
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

def test_duplicate_annotator_alias_error():
    """
    Test that adding two annotators with the same alias raises error immediately.
    This check happens at definition time (DSL), not validation time.
    """
    strategy = StringProducer()
    annotator = StringConsumer()
    
    with pytest.raises(PipelineConfigurationError) as excinfo:
        # Trying to register 'meta' twice
        Step(strategy) | annotate(annotator, alias="meta") | annotate(annotator, alias="meta")
        
    assert "Duplicate annotator alias 'meta'" in str(excinfo.value)

def test_duplicate_sibling_alias_error():
    """
    Test that two parallel steps cannot have the same alias.
    This check happens at validation time.
    """
    # Two steps with default alias "StringConsumer"
    s1 = Step(StringConsumer())
    s2 = Step(StringConsumer())
    
    pipeline = Step(StringProducer()) >> [s1, s2]
    
    with pytest.raises(PipelineConfigurationError) as excinfo:
        validate(pipeline)
        
    assert "Duplicate step alias 'StringConsumer'" in str(excinfo.value)

def test_sibling_alias_collision_resolution():
    """
    Test that providing explicit aliases resolves the collision.
    """
    s1 = Step(StringConsumer(), alias="v1")
    s2 = Step(StringConsumer(), alias="v2")
    
    pipeline = Step(StringProducer()) >> [s1, s2]
    
    # Should pass validation now
    validate(pipeline)