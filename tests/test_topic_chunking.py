import pytest
from typing import List, Dict, Any
from pumpking.models import ChunkPayload, TopicChunkPayload
from pumpking.protocols import ExecutionContext, TopicProviderProtocol
from pumpking.strategies.base import BaseStrategy
from pumpking.strategies.advanced import TopicBasedChunking

# --- Mocks ---

class MockAnnotator(BaseStrategy):
    """Spy annotator to verify execution count."""
    def __init__(self):
        self.call_count = 0
        
    def execute(self, data: str, context: ExecutionContext) -> dict:
        self.call_count += 1
        return {"checked": True}

class MockTopicProvider(TopicProviderProtocol):
    """Simulates topic assignment (Matrix: Chunks -> Topics)."""
    def __init__(self, matrix: List[List[str]]):
        self.matrix = matrix

    def assign_topics(self, chunks: List[str], **kwargs) -> List[List[str]]:
        return self.matrix

class MockSplitter(BaseStrategy):
    """Simulates ParagraphChunking with annotation support."""
    def execute(self, data: str, context: ExecutionContext) -> List[ChunkPayload]:
        texts = [t.strip() for t in data.split('\n\n') if t.strip()]
        payloads = []
        for t in texts:
            p = self._apply_annotators_to_payload(t, context)
            payloads.append(p)
        return payloads

# --- Tests ---

def test_topic_chunking_groups_correctly():
    """
    Verifies that chunks are routed to the correct Topic container based on the provider.
    """
    text = "Para1\n\nPara2\n\nPara3"
    
    matrix = [
        ["TopicA"],
        ["TopicB"],
        ["TopicA", "TopicB"]
    ]
    
    provider = MockTopicProvider(matrix)
    splitter = MockSplitter()
    strategy = TopicBasedChunking(topic_provider=provider, splitter=splitter)
    context = ExecutionContext()
    
    results = strategy.execute(text, context)
    
    assert len(results) == 2 
    
    topic_a = next(r for r in results if r.topic == "TopicA")
    assert len(topic_a.children) == 2
    assert topic_a.children[0].content == "Para1"
    assert topic_a.children[1].content == "Para3"
    
    topic_b = next(r for r in results if r.topic == "TopicB")
    assert len(topic_b.children) == 2
    assert topic_b.children[0].content == "Para2"
    assert topic_b.children[1].content == "Para3"

def test_topic_chunking_annotators_execution():
    """
    CRITICAL: Verifies that annotators are executed on the chunks 
    via the splitter delegation.
    """
    text = "Para1"
    matrix = [["TopicA"]]
    
    spy = MockAnnotator()
    context = ExecutionContext()
    context.annotators = {"spy": spy}
    
    strategy = TopicBasedChunking(
        topic_provider=MockTopicProvider(matrix),
        splitter=MockSplitter()
    )
    
    results = strategy.execute(text, context)
    
    topic_node = results[0]
    child_chunk = topic_node.children[0]
    
    assert "spy" in child_chunk.annotations
    assert child_chunk.annotations["spy"]["checked"] is True
    
    assert spy.call_count == 1

def test_topic_chunking_shared_references():
    """
    Verifies that if a chunk belongs to multiple topics, 
    it remains the same object in memory (Optimization).
    """
    text = "MultiTopicPara"
    matrix = [["TopicA", "TopicB"]] 
    
    strategy = TopicBasedChunking(
        topic_provider=MockTopicProvider(matrix),
        splitter=MockSplitter()
    )
    
    results = strategy.execute(text, ExecutionContext())
    
    topic_a = next(r for r in results if r.topic == "TopicA")
    topic_b = next(r for r in results if r.topic == "TopicB")
    
    chunk_in_a = topic_a.children[0]
    chunk_in_b = topic_b.children[0]
    
    assert chunk_in_a is chunk_in_b