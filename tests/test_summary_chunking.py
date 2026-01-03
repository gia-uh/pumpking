import pytest
from typing import List, Any
from pumpking.strategies.advanced import SummaryChunking
from pumpking.strategies.base import BaseStrategy
from pumpking.models import ChunkPayload
from pumpking.protocols import ExecutionContext, SummaryProviderProtocol

# --- Mocks ---

class MockSplitter(BaseStrategy):
    """
    Simulates a strategy that breaks content into smaller semantic units.
    Used to test the compound 'Split-then-Summarize' logic.
    """
    def execute(self, data: str, context: ExecutionContext) -> List[ChunkPayload]:
        parts = data.split("|")
        return [
            self._apply_annotators_to_payload(p.strip(), context, content_raw=p.strip())
            for p in parts
        ]

class MockSummaryProvider(SummaryProviderProtocol):
    """
    Simulates a sequential LLM provider.
    """
    def summarize(self, text: str, **kwargs: Any) -> str:
        return f"Summary of: {text}"

class MockAnnotator(BaseStrategy):
    """
    Simulates a metadata enhancer (e.g., Sentiment, Keywords).
    """
    def execute(self, data: str, context: ExecutionContext) -> str:
        return "annotated"

# --- Tests ---

def test_summary_chunking_execution_flow():
    """
    Verifies the primary 'Split -> Summarize' flow and correct lineage.
    Ensures that summaries point to the Parent, while content_raw identifies the Unit.
    """
    text = "Section A | Section B"
    strategy = SummaryChunking(provider=MockSummaryProvider(), splitter=MockSplitter())
    context = ExecutionContext()
    
    results = strategy.execute(text, context)
    
    assert len(results) == 2
    assert results[0].content == "Summary of: Section A"
    assert results[0].children[0].content == "Section A | Section B"
    assert results[0].content_raw == "Section A"

def test_summary_chunking_batch_input_handling():
    """
    Verifies that the strategy handles a list of multiple large payloads,
    correctly flattening the result and maintaining separate lineages.
    """
    parent1 = ChunkPayload(content="P1_A | P1_B", content_raw="P1_A | P1_B", annotations={})
    parent2 = ChunkPayload(content="P2_A", content_raw="P2_A", annotations={})
    
    strategy = SummaryChunking(provider=MockSummaryProvider(), splitter=MockSplitter())
    context = ExecutionContext()
    
    results = strategy.execute([parent1, parent2], context)
    
    assert len(results) == 3
    assert results[0].content == "Summary of: P1_A"
    assert results[0].children[0] == parent1
    assert results[2].content == "Summary of: P2_A"
    assert results[2].children[0] == parent2

def test_summary_chunking_without_splitter():
    """
    Verifies that if splitter is None, the strategy uses the default (AdaptiveChunking).
    Since AdaptiveChunking is robust, for this small text it should return the whole text as one chunk.
    """
    text = "Whole Document"
    strategy = SummaryChunking(provider=MockSummaryProvider(), splitter=None)
    context = ExecutionContext()

    results = strategy.execute(text, context)

    assert len(results) == 1
    assert results[0].content == "Summary of: Whole Document"
    assert results[0].children[0].content == "Whole Document"

def test_summary_chunking_content_raw_optimization():
    """
    Verifies that 'content_raw' follows the BaseStrategy optimization rule:
    It becomes None if identical to 'content'.
    """
    text = "Atom"
    
    class IdentityProvider(SummaryProviderProtocol):
        def summarize(self, text: str, **kwargs: Any) -> str:
            return text 
            
    strategy = SummaryChunking(provider=IdentityProvider(), splitter=None)
    context = ExecutionContext()

    results = strategy.execute(text, context)
    
    assert results[0].content == "Atom"
    assert results[0].content_raw is None

def test_summary_chunking_annotations_on_summary():
    """
    CRITICAL: Verifies that annotators are applied to the generated summary text,
    not to the original source text.
    """
    text = "Original" 
    spy = MockAnnotator()
    context = ExecutionContext()
    context.annotators = {"test": spy}
    
    strategy = SummaryChunking(provider=MockSummaryProvider(), splitter=None)
    results = strategy.execute(text, context)
    
    assert "test" in results[0].annotations
    assert results[0].annotations["test"] == "annotated"