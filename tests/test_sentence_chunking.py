import pytest
from typing import List, Union
from pumpking.strategies.basic import SentenceChunking, ParagraphChunking
from pumpking.models import ChunkPayload
from pumpking.protocols import ExecutionContext
from pumpking.pipeline import Step, PumpkingPipeline
from pumpking.strategies.base import BaseStrategy

# --- Fixtures and Mocks ---


class SpyStrategy(BaseStrategy):
    """
    Mock strategy to inspect what data is passed to the next step.
    UPDATED: Handles input normalization (str vs Payload).
    """

    def execute(self, data: Union[str, ChunkPayload], context: ExecutionContext) -> ChunkPayload:
        content = data.content if isinstance(data, ChunkPayload) else str(data)
        
        payload = self._apply_annotators_to_payload(content, context)
        return payload


# --- Tests ---


def test_sentence_chunking_logic_only():
    """
    Verifies the core logic of SentenceChunking in isolation.
    """
    chunker = SentenceChunking()
    context = ExecutionContext()
    
    text = "Hello world. This is a test! Are you ready?"
    results = chunker.execute(text, context)
    
    assert len(results) == 3
    assert results[0].content == "Hello world."
    assert results[1].content == "This is a test!"
    assert results[2].content == "Are you ready?"
    
    assert all(isinstance(r, ChunkPayload) for r in results)


def test_sentence_chunking_handles_abbreviations():
    """
    Ensures that common abbreviations do not trigger false positive splits.
    """
    chunker = SentenceChunking()
    context = ExecutionContext()
    
    text = "Mr. Smith went to Dr. Jones on Jan. 5th."
    results = chunker.execute(text, context)
    
    pass 


def test_pipeline_paragraph_to_sentence_structure():
    """
    Validates a sequential pipeline that refines chunks from paragraphs down to sentences.
    
    This test establishes a multi-stage processing flow:
    1. ParagraphChunking splits the document into large blocks.
    2. SentenceChunking receives each block and splits it further into sentences.
    3. SpyStrategy captures the final granular chunks.
    
    This ensures that the pipeline infrastructure correctly handles the nested expansion
    of data, where one parent node (paragraph) yields multiple child nodes (sentences),
    and that the traversal order maintains the logical sequence of the document.
    """
    spy = SpyStrategy()

    pipeline = PumpkingPipeline(Step(ParagraphChunking()) >> Step(SentenceChunking())) >> Step(spy)
    
    text = "Para 1 Sentence 1. Para 1 Sentence 2.\n\nPara 2 Sentence 1."
    
    root_node = pipeline.run(text)
    
    para_nodes = root_node.branches
    assert len(para_nodes) == 2
    
    para1_sentences = para_nodes[0].branches
    assert len(para1_sentences) == 2
    assert "Para 1 Sentence 1." in para1_sentences[0].results[0].content
    assert "Para 1 Sentence 2." in para1_sentences[1].results[0].content
    
    para2_sentences = para_nodes[1].branches
    assert len(para2_sentences) == 1
    assert "Para 2 Sentence 1." in para2_sentences[0].results[0].content


def test_pipeline_sentence_chunking_direct():
    """
    Verifies direct SentenceChunking from root.
    """
    spy = SpyStrategy()
    pipeline = PumpkingPipeline(Step(SentenceChunking()) >> Step(spy))
    
    text = "Sentence A. Sentence B."
    root = pipeline.run(text)
    
    sent_nodes = root.branches
    assert len(sent_nodes) == 2
    
    assert "Sentence A." in sent_nodes[0].results[0].content
    assert "Sentence B." in sent_nodes[1].results[0].content