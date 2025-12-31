import pytest
from typing import List, Any, Union
from pumpking.models import ChunkPayload, EntityChunkPayload, NERResult
from pumpking.protocols import ExecutionContext, NERProviderProtocol
from pumpking.strategies.base import BaseStrategy
from pumpking.strategies.advanced import EntityBasedChunking

# --- Mocks for Testing ---

class MockAnnotator(BaseStrategy):
    """
    A spy annotator to verify that annotations are computed only once
    and preserved across shared references.
    """
    def __init__(self):
        self.call_count = 0
        
    def execute(self, data: str, context: ExecutionContext) -> dict:
        self.call_count += 1
        return {"spy_checked": True, "id": self.call_count}

class MockNERProvider(NERProviderProtocol):
    """
    Simulates the LLMProvider behavior by returning pre-defined indices.
    This avoids making real API calls during tests.
    """
    def __init__(self, fixed_results: List[NERResult]):
        self.fixed_results = fixed_results

    def extract_entities(self, sentences: List[str], **kwargs: Any) -> List[NERResult]:
        return self.fixed_results

class MockSplitter(BaseStrategy):
    """
    Simulates SentenceChunking.
    Splits text by periods for simplicity in testing.
    """
    def execute(self, data: str, context: ExecutionContext) -> List[ChunkPayload]:
        segments = [s.strip() for s in data.split('.') if s.strip()]
        payloads = []
        for seg in segments:
            p = ChunkPayload(content=seg, content_raw=seg)
            p = self._apply_annotators_to_payload(p.content, context, p.content_raw)
            payloads.append(p)
        return payloads

# --- Tests ---

def test_entity_chunking_groups_correctly():
    """
    Verifies that the strategy correctly groups sentences under their respective entities
    based on the indices provided by the NER provider.
    """
    
    mock_results = [
        NERResult(entity="Elon Musk", label="PER", indices=[0, 1]),
        NERResult(entity="SpaceX", label="ORG", indices=[0]),
        NERResult(entity="Tesla", label="ORG", indices=[1]),
    ]
    
    provider = MockNERProvider(mock_results)
    splitter = MockSplitter()
    
    strategy = EntityBasedChunking(ner_provider=provider, splitter=splitter)
    context = ExecutionContext()
    
    text = "Elon Musk runs SpaceX. He runs Tesla."
    
    results = strategy.execute(text, context)
    
    assert len(results) == 3 
    
    elon_node = next(r for r in results if isinstance(r, EntityChunkPayload) and r.entity == "Elon Musk")
    assert elon_node.type == "PER"
    assert len(elon_node.children) == 2
    assert "Elon Musk runs SpaceX" in elon_node.children[0].content
    assert "He runs Tesla" in elon_node.children[1].content
    
    spacex_node = next(r for r in results if isinstance(r, EntityChunkPayload) and r.entity == "SpaceX")
    assert len(spacex_node.children) == 1
    assert "Elon Musk runs SpaceX" in spacex_node.children[0].content

def test_entity_chunking_optimization_shared_memory():
    """
    CRITICAL TEST: Verifies the reference sharing optimization.
    
    If Sentence A belongs to Entity X and Entity Y, both EntityChunkPayloads
    must point to the EXACT SAME ChunkPayload object in memory for Sentence A.
    This proves we are not duplicating data or re-running annotations.
    """
    mock_results = [
        NERResult(entity="Ent1", label="MISC", indices=[0]),
        NERResult(entity="Ent2", label="MISC", indices=[0]),
    ]
    
    provider = MockNERProvider(mock_results)
    splitter = MockSplitter()
    strategy = EntityBasedChunking(ner_provider=provider, splitter=splitter)
    context = ExecutionContext()
    
    text = "Shared Sentence."
    results = strategy.execute(text, context)
    
    node1 = results[0] 
    node2 = results[1] 
    
    child1 = node1.children[0]
    child2 = node2.children[0]
    
    assert child1 is child2
    assert child1.content == "Shared Sentence"

def test_entity_chunking_annotators_run_once():
    """
    Verifies that annotators are executed during the splitting phase 
    and NOT re-executed during the grouping phase.
    """
    mock_results = [
        NERResult(entity="Ent1", label="MISC", indices=[0]),
        NERResult(entity="Ent2", label="MISC", indices=[0]),
    ]
    
    spy_annotator = MockAnnotator()
    context = ExecutionContext()
    context.annotators = {"spy": spy_annotator}
    
    provider = MockNERProvider(mock_results)
    splitter = MockSplitter()
    strategy = EntityBasedChunking(ner_provider=provider, splitter=splitter)
    
    text = "Single Sentence."
    
    results = strategy.execute(text, context)
    
    child = results[0].children[0]
    
    assert "spy" in child.annotations
    assert child.annotations["spy"]["spy_checked"] is True
    
    assert spy_annotator.call_count == 1

def test_entity_chunking_default_instantiation():
    """
    Verifies that the class correctly handles Class Types passed as arguments
    instead of instances.
    """
    class SimpleMockProvider(NERProviderProtocol):
        def extract_entities(self, sentences, **kwargs):
            return []
            
    strategy = EntityBasedChunking(
        ner_provider=SimpleMockProvider, 
        splitter=MockSplitter            
    )
    
    assert isinstance(strategy.ner_provider, SimpleMockProvider)
    assert isinstance(strategy.splitter, MockSplitter)

def test_entity_chunking_empty_input():
    """
    Verifies graceful handling of empty strings.
    """
    strategy = EntityBasedChunking(
        ner_provider=MockNERProvider([]), 
        splitter=MockSplitter()
    )
    context = ExecutionContext()
    
    assert strategy.execute("", context) == []
    assert strategy.execute(None, context) == []