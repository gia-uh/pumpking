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
            ctx_str = f"Context for {chunk.content}"
            
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
        return [self._apply_annotators_to_payload(s, context) for s in segments]

class MockMismatchProvider(ContextualProviderProtocol):
    """
    Simulates a broken/filtering provider returning fewer contexts than chunks.
    """
    def assign_context(self, chunks: List[ChunkPayload], **kwargs: Any) -> List[ContextualChunkPayload]:
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
    
    assert results[0].content == "Block A"
    assert results[0].context == "Context for Block A"
    assert len(results[0].children) == 1
    
    assert results[1].content == "Block B"
    assert results[1].context == "Context for Block B"

def test_contextual_chunking_annotations_lifecycle():
    """
    Verifies the annotation lifecycle:
    1. Splitter annotates the 'content' (Original Text) once.
    2. Strategy DOES NOT re-annotate content nor the generated 'context'.
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
    
    assert len(spy.call_history) == 1
    assert spy.call_history[0] == "Fragment"
    
    assert "Context for Fragment" not in spy.call_history
    
    payload = results[0]
    assert payload.annotations["spy"]["checked"] is True
    

def test_contextual_chunking_batch_delegation():
    """
    Verifies that the strategy delegates the full list of extracted payloads
    to the provider in a single call, passing provider_kwargs correctly.
    """
    class SpyProvider(ContextualProviderProtocol):
        def __init__(self):
            self.received_chunks = []
            self.received_kwargs = {}
            
        def assign_context(self, chunks: List[ChunkPayload], **kwargs: Any) -> List[ContextualChunkPayload]:
            self.received_chunks = chunks
            self.received_kwargs = kwargs
            return []

    data = "A | B | C"
    spy_provider = SpyProvider()
    
    strategy = ContextualChunking(
        provider=spy_provider,
        splitter=MockSplitter(),
        model="gpt-4-turbo"
    )
    
    strategy.execute(data, ExecutionContext())
    
    assert len(spy_provider.received_chunks) == 3
    assert spy_provider.received_chunks[0].content == "A"
    assert spy_provider.received_chunks[2].content == "C"
    
    assert spy_provider.received_kwargs.get("model") == "gpt-4-turbo"

def test_contextual_chunking_alignment_delegation():
    """
    Verifies that the strategy respects the Provider's output structure.
    If the provider filters or fails to align (returns fewer items), 
    the strategy simply returns what was given (Architecture decision: Smart Provider).
    """
    data = "A | B | C"
    
    strategy = ContextualChunking(
        provider=MockMismatchProvider(),
        splitter=MockSplitter()
    )
    
    results = strategy.execute(data, ExecutionContext())
    
    assert len(results) == 1
    
    assert results[0].content == "A"
    assert results[0].context == "Only one context"