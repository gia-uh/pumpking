from typing import Any, List
from pumpking.protocols import ExecutionContext, StrategyProtocol
from pumpking.pipeline import Step, PumpkingPipeline, annotate

# --- Mocks ---

class MockStrategy:
    """Generic mock strategy."""
    SUPPORTED_INPUTS: List[Any] = [str]
    PRODUCED_OUTPUT: Any = str

    def execute(self, data: Any, context: ExecutionContext) -> Any:
        return data

class SentimentAnalysis(MockStrategy):
    """Named mock for testing class-name inference."""
    pass

class NamedEntityRecognition(MockStrategy):
    """Another named mock."""
    pass

# --- Tests for Step & Alias ---

def test_step_alias_defaults():
    """
    Case: Step(Strategy())
    Expectation: alias should be 'StrategyClass'.
    """
    step = Step(SentimentAnalysis())
    assert step.alias == "SentimentAnalysis"

def test_step_alias_explicit():
    """
    Case: Step(Strategy(), alias="my_step")
    Expectation: alias should be 'my_step'.
    """
    step = Step(SentimentAnalysis(), alias="custom_sentiment")
    assert step.alias == "custom_sentiment"

# --- Tests for Annotation Helper ---

def test_annotate_alias_defaults():
    """
    Case: Step | annotate(Strategy())
    Expectation: annotator key should be 'StrategyClass'.
    """
    step = Step(MockStrategy()) | annotate(SentimentAnalysis())
    
    assert "SentimentAnalysis" in step.annotators
    assert isinstance(step.annotators["SentimentAnalysis"], SentimentAnalysis)

def test_annotate_alias_explicit():
    """
    Case: Step | annotate(Strategy(), alias="my_anno")
    Expectation: annotator key should be 'my_anno'.
    """
    step = Step(MockStrategy()) | annotate(SentimentAnalysis(), alias="version_2")
    
    assert "version_2" in step.annotators
    assert "SentimentAnalysis" not in step.annotators

# --- Tests for Pipeline Structure ---

def test_linear_pipeline():
    """
    Case: Step >> Step
    """
    s1 = Step(MockStrategy(), alias="start")
    s2 = Step(MockStrategy(), alias="end")
    
    pipeline = s1 >> s2
    
    assert len(pipeline.steps) == 2
    assert pipeline.steps[0].alias == "start"
    assert pipeline.steps[1].alias == "end"

def test_parallel_pipeline_structure():
    """
    Case: Step >> [Step, Step]
    """
    start = Step(MockStrategy())
    branch_a = Step(SentimentAnalysis())
    branch_b = Step(NamedEntityRecognition())
    
    pipeline = start >> [branch_a, branch_b]
    
    # Structure verification
    assert len(pipeline.steps) == 2
    
    # Second step should be a list (Parallel Block)
    parallel_block = pipeline.steps[1]
    assert isinstance(parallel_block, list)
    assert len(parallel_block) == 2
    
    # Verify branches
    assert parallel_block[0].alias == "SentimentAnalysis"
    assert parallel_block[1].alias == "NamedEntityRecognition"

def test_complex_mixed_structure():
    """
    Case: Step >> [ Step | annotate, Step ]
    Verifies that annotations work inside parallel blocks.
    """
    # Branch 1: Sentiment + Annotation
    branch_1 = Step(SentimentAnalysis()) | annotate(NamedEntityRecognition(), alias="ner_check")
    
    # Branch 2: Just NER
    branch_2 = Step(NamedEntityRecognition())
    
    pipeline = Step(MockStrategy()) >> [branch_1, branch_2]
    
    parallel_block = pipeline.steps[1]
    
    # Check Branch 1 annotations
    b1_node = parallel_block[0]
    assert b1_node.alias == "SentimentAnalysis"
    assert "ner_check" in b1_node.annotators
    
    # Check Branch 2 annotations (should be empty)
    b2_node = parallel_block[1]
    assert b2_node.alias == "NamedEntityRecognition"
    assert b2_node.annotators == {}

def test_chaining_syntax():
    """
    Case: (Step >> Step) >> Step
    Ensures the pipeline object itself supports '>>'.
    """
    p = Step(MockStrategy()) >> Step(MockStrategy())
    # Now extend the existing pipeline
    p = p >> Step(MockStrategy())
    
    assert len(p.steps) == 3