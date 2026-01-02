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
        self.call_history = []

    def execute(self, data: str, context: ExecutionContext) -> dict:
        self.call_history.append(data)
        return {"checked": True}

class MockContextualProvider(ContextualProviderProtocol):
    """
    Simulates a provider that generates context based on the input list.
    Updated to return ContextualChunkPayloads as per the new protocol.
    """
    def assign_context(self, chunks: List[ChunkPayload], **kwargs: Any) -> List[ContextualChunkPayload]:
        results = []
        for chunk in chunks:
            # Simulate context generation
            ctx_str = f"Context for {chunk.content}"
            
            # Construct the full payload (Provider responsibility now)
            payload = ContextualChunkPayload(
                content=chunk.content,
                content_raw=chunk.content_raw,
                context=ctx_str,
                children=[chunk],
                annotations=chunk.annotations.copy()
            )
            results.append(payload)
        return results

class MockSplitter(BaseStrategy):
    """Splits by pipe '|' for deterministic testing."""
    def execute(self, data: str, context: ExecutionContext) -> List[ChunkPayload]:
        segments = [s.strip() for s in data.split('|') if s.strip()]
        # Annotators applied to CONTENT during splitting
        return [self._apply_annotators_to_payload(s, context) for s in segments]

class MockMismatchProvider(ContextualProviderProtocol):
    """
    Simulates a broken/filtering provider returning fewer contexts than chunks.
    """
    def assign_context(self, chunks: List[ChunkPayload], **kwargs: Any) -> List[ContextualChunkPayload]:
        # Returns only 1 item regardless of input size
        if not chunks:
            return []
        
        first = chunks[0]
        return [
            ContextualChunkPayload(
                content=first.content, 
                context="Only one context",
                children=[first]
            )
        ]

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
    assert len(results[0].children) == 1
    
    # Item 1
    assert results[1].content == "Block B"
    assert results[1].context == "Context for Block B"

def test_contextual_chunking_annotations_lifecycle():
    """
    Verifies the full annotation lifecycle in the new architecture:
    1. Splitter annotates the 'content' (Original Text).
    2. Strategy annotates the 'context' (Generated Text).
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
    
    payload = results[0]
    
    # 1. Verify Annotator ran twice (Once for Content, Once for Context)
    assert len(spy.call_history) == 2
    assert "Fragment" in spy.call_history            # Phase 1: Splitter
    assert "Context for Fragment" in spy.call_history # Phase 2: Strategy
    
    # 2. Verify Annotations are present in the final payload
    # Note: Depending on implementation, they might merge or be separate. 
    # Usually, the payload carries the annotations from the splitter + new ones.
    assert "spy" in payload.annotations

def test_contextual_chunking_batch_delegation():
    """
    Verifies that the strategy delegates the full list of extracted payloads
    to the provider in a single call, passing provider_kwargs correctly.
    """
    # Specialized mock to capture arguments
    class SpyProvider(ContextualProviderProtocol):
        def __init__(self):
            self.received_chunks = []
            self.received_kwargs = {}
            
        def assign_context(self, chunks: List[ChunkPayload], **kwargs: Any) -> List[ContextualChunkPayload]:
            self.received_chunks = chunks
            self.received_kwargs = kwargs
            # Return empty list to satisfy type hint, content irrelevant for this test
            return []

    data = "A | B | C"
    spy_provider = SpyProvider()
    
    strategy = ContextualChunking(
        provider=spy_provider,
        splitter=MockSplitter(),
        model="gpt-4-turbo" # Kwarg to verify
    )
    
    strategy.execute(data, ExecutionContext())
    
    # Verify the provider received the exact list of ChunkPayload objects
    assert len(spy_provider.received_chunks) == 3
    assert spy_provider.received_chunks[0].content == "A"
    assert spy_provider.received_chunks[2].content == "C"
    
    # Verify kwargs were passed through
    assert spy_provider.received_kwargs.get("model") == "gpt-4-turbo"

def test_contextual_chunking_alignment_delegation():
    """
    Verifies that the strategy respects the Provider's output structure.
    If the provider filters or fails to align (returns fewer items), 
    the strategy simply returns what was given (Architecture decision: Smart Provider).
    """
    data = "A | B | C"
    
    # Provider returns 1 context for 3 chunks
    strategy = ContextualChunking(
        provider=MockMismatchProvider(),
        splitter=MockSplitter()
    )
    
    results = strategy.execute(data, ExecutionContext())
    
    # The Strategy no longer attempts to force padding. 
    # It trusts the Provider to return the correct list of payloads.
    assert len(results) == 1
    
    # Verify the one surviving payload
    assert results[0].content == "A"
    assert results[0].context == "Only one context"