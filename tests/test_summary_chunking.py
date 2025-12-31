import pytest
from typing import List, Any
from pumpking.models import ChunkPayload
from pumpking.protocols import ExecutionContext, SummaryProviderProtocol
from pumpking.strategies.base import BaseStrategy
from pumpking.strategies.advanced import SummaryChunking

# --- Mocks ---

class MockAnnotator(BaseStrategy):
    """Spy annotator to verify which text is being processed."""
    def __init__(self):
        self.processed_text = None
        
    def execute(self, data: str, context: ExecutionContext) -> dict:
        self.processed_text = data
        return {"checked": True}

class MockSummaryProvider(SummaryProviderProtocol):
    """Simulates summarization by prefixing text."""
    def summarize(self, text: str, **kwargs: Any) -> str:
        return f"Summary of: {text}"

class MockSplitter(BaseStrategy):
    """Simulates splitting by creating predefined payloads."""
    def execute(self, data: str, context: ExecutionContext) -> List[ChunkPayload]:
        segments = [s.strip() for s in data.split('|') if s.strip()]
        return [self._apply_annotators_to_payload(s, context) for s in segments]

# --- Tests ---

def test_summary_chunking_payload_structure():
    """
    Verifies that the output payload correctly places the summary in 'content'
    and the original text in 'content_raw'.
    """
    text = "Section A | Section B"
    
    provider = MockSummaryProvider()
    splitter = MockSplitter()
    strategy = SummaryChunking(provider=provider, splitter=splitter)
    context = ExecutionContext()
    
    results = strategy.execute(text, context)
    
    assert len(results) == 2
    
    payload_a = results[0]
    assert payload_a.content == "Summary of: Section A"
    assert payload_a.content_raw == "Section A"
    
    payload_b = results[1]
    assert payload_b.content == "Summary of: Section B"
    assert payload_b.content_raw == "Section B"

def test_summary_chunking_annotations_target_summary():
    """
    CRITICAL: Verifies that the annotators are executed against the generated
    summary, not the original raw text.
    """
    text = "Complex Legal Text"
    expected_summary = "Summary of: Complex Legal Text"
    
    spy = MockAnnotator()
    context = ExecutionContext()
    context.annotators = {"spy": spy}
    
    strategy = SummaryChunking(
        provider=MockSummaryProvider(), 
        splitter=MockSplitter()
    )
    
    results = strategy.execute(text, context)
    
    assert "spy" in results[0].annotations
    
    assert spy.processed_text == expected_summary
    assert spy.processed_text != text

def test_summary_chunking_empty_input():
    """Verifies that empty input returns an empty list without errors."""
    strategy = SummaryChunking(
        provider=MockSummaryProvider(), 
        splitter=MockSplitter()
    )
    assert strategy.execute("", ExecutionContext()) == []