import pytest
from typing import List, Dict, Any
from pumpking.models import ChunkPayload, TopicChunkPayload
from pumpking.protocols import ExecutionContext, TopicProviderProtocol
from pumpking.strategies.base import BaseStrategy
from pumpking.strategies.advanced import TopicBasedChunking

# --- Mocks ---

class MockAnnotator(BaseStrategy):
    def __init__(self):
        self.call_count = 0
    def execute(self, data, context):
        self.call_count += 1
        return {"checked": True}

class MockTopicProvider(TopicProviderProtocol):
    """
    Simulates topic assignment (Matrix: Chunks -> Topics).
    Crucial: Implements 'assign_topics' accepting chunks and returning grouped TopicPayloads.
    """
    def __init__(self, matrix: List[List[str]]):
        self.matrix = matrix

    def assign_topics(self, chunks: List[ChunkPayload], **kwargs) -> List[TopicChunkPayload]:
        topic_map: Dict[str, List[ChunkPayload]] = {}
        
        limit = min(len(chunks), len(self.matrix))
        for i in range(limit):
            chunk = chunks[i]
            topics = self.matrix[i]
            for t in topics:
                if t not in topic_map:
                    topic_map[t] = []
                topic_map[t].append(chunk)
        
        results = []
        for topic, children in topic_map.items():
            results.append(TopicChunkPayload(
                topic=topic, content=topic, children=children
            ))
        return results

class MockSplitter(BaseStrategy):
    """Simulates ParagraphChunking with annotation support."""
    def execute(self, data: str, context: ExecutionContext) -> List[ChunkPayload]:
        texts = [t.strip() for t in data.split('\n\n') if t.strip()]
        return [self._apply_annotators_to_payload(t, context) for t in texts]

# --- Tests ---

def test_topic_chunking_groups_correctly():
    text = "Para1\n\nPara2\n\nPara3"
    matrix = [["TopicA"], ["TopicB"], ["TopicA", "TopicB"]] 
    
    provider = MockTopicProvider(matrix)
    strategy = TopicBasedChunking(topic_provider=provider, splitter=MockSplitter())
    
    results = strategy.execute(text, ExecutionContext())
    
    assert len(results) == 2 
    
    topic_a = next(r for r in results if r.topic == "TopicA")
    assert len(topic_a.children) == 2 
    assert topic_a.children[0].content == "Para1"
    
    topic_b = next(r for r in results if r.topic == "TopicB")
    assert len(topic_b.children) == 2 

def test_topic_chunking_annotators_execution():
    text = "Para1"
    matrix = [["TopicA"]]
    spy = MockAnnotator()
    ctx = ExecutionContext(annotators={"spy": spy})
    
    strategy = TopicBasedChunking(topic_provider=MockTopicProvider(matrix), splitter=MockSplitter())
    results = strategy.execute(text, ctx)
    
    child = results[0].children[0]
    assert child.annotations["spy"]["checked"] is True
    
    topic_node = results[0]
    assert topic_node.annotations["spy"]["checked"] is True
    
    assert spy.call_count == 2

def test_topic_chunking_shared_references():
    text = "MultiTopicPara"
    matrix = [["TopicA", "TopicB"]]
    strategy = TopicBasedChunking(topic_provider=MockTopicProvider(matrix), splitter=MockSplitter())
    
    results = strategy.execute(text, ExecutionContext())
    topic_a = next(r for r in results if r.topic == "TopicA")
    topic_b = next(r for r in results if r.topic == "TopicB")
    
    assert topic_a.children[0] is topic_b.children[0]