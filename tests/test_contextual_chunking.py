import pytest
from typing import List, Any
from pumpking.models import ContextualChunkPayload, ChunkPayload
from pumpking.protocols import ExecutionContext, ContextualProviderProtocol
from pumpking.strategies.base import BaseStrategy
from pumpking.strategies.advanced import ContextualChunking

# --- Mocks ---

class MockAnnotator(BaseStrategy):
    """Spy annotator to verify execution on specific content."""
    def __init__(self):
        self.last_content = None

    def execute(self, data: str, context: ExecutionContext) -> dict:
        self.last_content = data
        return {"checked": True}

class MockContextualProvider(ContextualProviderProtocol):
    """
    Simulates a provider that generates context based on the input list.
    """
    def assign_context(self, chunks: List[str], **kwargs: Any) -> List[str]:
        # Returns a mock context for each chunk, e.g., "Context for [Original]"
        return [f"Context for {c}" for c in chunks]

class MockSplitter(BaseStrategy):
    """Splits by pipe '|' for deterministic testing."""
    def execute(self, data: str, context: ExecutionContext) -> List[ChunkPayload]:
        segments = [s.strip() for s in data.split('|') if s.strip()]
        return [self._apply_annotators_to_payload(s, context) for s in segments]

class MockMismatchProvider(ContextualProviderProtocol):
    """Simulates a broken provider returning fewer contexts than chunks."""
    def assign_context(self, chunks: List[str], **kwargs: Any) -> List[str]:
        return ["Only one context"] # Returns only 1 item regardless of input size

# --- Tests ---

def test_contextual_chunking_structure():
    """
    Verifies that the strategy correctly assembles the ContextualChunkPayload,
    placing the provider's output in 'context' and the original text in 'content'.
    """
    data = "Block A | Block B"
    
    strategy = ContextualChunking(
        provider=MockContextualProvider(),
        splitter=MockSplitter()
    )
    context = ExecutionContext()
    
    results = strategy.execute(data, context)
    
    assert len(results) == 2
    assert isinstance(results[0], ContextualChunkPayload)
    
    # Check alignment
    # Item 0
    assert results[0].content == "Block A"
    assert results[0].context == "Context for Block A"
    
    # Item 1
    assert results[1].content == "Block B"
    assert results[1].context == "Context for Block B"

def test_contextual_chunking_annotations_on_content():
    """
    Verifies that annotators are applied to the 'content' (the text fragment),
    not the generated context string.
    """
    data = "Fragment"
    
    spy = MockAnnotator()
    ctx = ExecutionContext()
    ctx.annotators = {"spy": spy}
    
    strategy = ContextualChunking(
        provider=MockContextualProvider(),
        splitter=MockSplitter()
    )
    
    results = strategy.execute(data, ctx)
    
    assert len(results) == 1
    payload = results[0]
    
    # Verify annotation is present
    assert "spy" in payload.annotations
    
    # Verify annotator saw the content, not the context
    assert spy.last_content == "Fragment"
    assert spy.last_content != "Context for Fragment"

def test_contextual_chunking_batch_delegation():
    """
    Verifies that the strategy delegates the full list of extracted texts
    to the provider in a single call (or aligned calls), passing the
    provider_kwargs correctly.
    """
    # We define a specialized mock to capture arguments
    class SpyProvider(ContextualProviderProtocol):
        def __init__(self):
            self.received_chunks = []
            self.received_kwargs = {}
            
        def assign_context(self, chunks: List[str], **kwargs: Any) -> List[str]:
            self.received_chunks = chunks
            self.received_kwargs = kwargs
            return [""] * len(chunks)

    data = "A | B | C"
    spy_provider = SpyProvider()
    
    strategy = ContextualChunking(
        provider=spy_provider,
        splitter=MockSplitter(),
        model="gpt-4-turbo" # Kwarg to verify
    )
    
    strategy.execute(data, ExecutionContext())
    
    # Verify the provider received the exact list of contents
    assert spy_provider.received_chunks == ["A", "B", "C"]
    
    # Verify kwargs were passed through
    assert spy_provider.received_kwargs.get("model") == "gpt-4-turbo"

def test_contextual_chunking_alignment_safety():
    """
    Verifies that if the provider returns fewer contexts than chunks,
    the strategy handles alignment safely (e.g., by padding/truncating)
    instead of crashing or misaligning.
    """
    data = "A | B | C"
    
    # Provider returns 1 context for 3 chunks
    strategy = ContextualChunking(
        provider=MockMismatchProvider(),
        splitter=MockSplitter()
    )
    
    results = strategy.execute(data, ExecutionContext())
    
    assert len(results) == 3
    
    # First chunk got the context
    assert results[0].content == "A"
    assert results[0].context == "Only one context"
    
    # Subsequent chunks should handle missing context gracefully (empty string or similar)
    # Based on the strategy implementation which pads with ""
    assert results[1].content == "B"
    assert results[1].context == ""
    assert results[2].content == "C"
    assert results[2].context == ""