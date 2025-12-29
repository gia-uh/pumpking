import pytest

from typing import List, Any
from pumpking.models import ChunkPayload
from pumpking.pipeline import annotate, Step
from pumpking.protocols import ExecutionContext
from pumpking.strategies.base import BaseStrategy
from pumpking.strategies.basic import (
    RegexChunking,
    FixedSizeChunking,
    ParagraphChunking,
    SentenceChunking,
    SlidingWindowChunking
)

COMPLEX_MARKDOWN = """# System Architecture

The system is built using a **microservices** architecture.

## Core Components

1. **API Gateway**: Handles incoming requests.
2. **Auth Service**: Manages user identity.
   * Supports OAuth2.
   * Supports JWT.

## Data Flow

```json
{
  "source": "Client",
  "destination": "Server"
}

Future Roadmap

> Warning: This module is deprecated. """


class SpyStrategy(BaseStrategy):
    SUPPORTED_INPUTS = [str]
    PRODUCED_OUTPUT = str

    def __init__(self):
        self.received_chunks: List[str] = []

    def execute(self, data: str, context: ExecutionContext) -> str:
        self.received_chunks.append(data)
        return data


def test_regex_chunking_strategy_pure_logic():
    strategy = RegexChunking(pattern=r"\n\n+")
    context = ExecutionContext()
    payloads = strategy.execute(COMPLEX_MARKDOWN, context)

    print(payloads)

    assert isinstance(payloads, list)
    assert len(payloads) >= 6
    assert isinstance(payloads[0], ChunkPayload)

    assert payloads[0].content == "# System Architecture"

    assert "microservices" in payloads[1].content
    assert payloads[2].content == "## Core Components"
    assert "1. **API Gateway**" in payloads[3].content
    assert "* Supports OAuth2." in payloads[3].content
    assert payloads[4].content == "## Data Flow"
    assert "```json" in payloads[5].content


def test_regex_chunking_pipeline_integration_flow():
    chunker = RegexChunking(pattern=r"\n\n+")
    spy = SpyStrategy()

    pipeline = Step(chunker) >> Step(spy)

    root_node = pipeline.run(COMPLEX_MARKDOWN)

    assert root_node.content == COMPLEX_MARKDOWN

    assert len(spy.received_chunks) >= 6
    assert "# System Architecture" in spy.received_chunks
    assert "## Core Components" in spy.received_chunks

    assert spy.received_chunks[0] == "# System Architecture"
    assert "microservices" in spy.received_chunks[1]


def test_fixed_size_chunking_cuts_structure_blindly():
    chunk_size = 20
    strategy = FixedSizeChunking(chunk_size=chunk_size, overlap=0)
    context = ExecutionContext()

    payloads = strategy.execute(COMPLEX_MARKDOWN, context)

    assert len(payloads) > 0
    assert payloads[0].content == "# System Architectur"

    assert payloads[1].content == "e The system is bui"

    assert payloads[1].content_raw == "e\n\nThe system is bui"


def test_fixed_size_chunking_overlap_consistency():
    chunk_size = 50
    overlap = 10
    strategy = FixedSizeChunking(chunk_size=chunk_size, overlap=overlap)
    context = ExecutionContext()

    payloads = strategy.execute(COMPLEX_MARKDOWN, context)

    first_chunk_end = payloads[0].content_raw[-overlap:]
    second_chunk_start = payloads[1].content_raw[:overlap]

    assert first_chunk_end == second_chunk_start


def test_fixed_size_chunking_preserves_total_content():
    chunk_size = 100
    strategy = FixedSizeChunking(chunk_size=chunk_size, overlap=0)
    context = ExecutionContext()

    payloads = strategy.execute(COMPLEX_MARKDOWN, context)

    reconstructed = "".join([p.content_raw for p in payloads])
    assert reconstructed == COMPLEX_MARKDOWN


def test_fixed_size_chunking_validation_error():
    with pytest.raises(ValueError):
        FixedSizeChunking(chunk_size=0)

    with pytest.raises(ValueError):
        FixedSizeChunking(chunk_size=5, overlap=6)


def test_paragraph_chunking_splits_on_double_newlines():
    strategy = ParagraphChunking()
    context = ExecutionContext()

    text = "Para 1.\nStill Para 1.\n\nPara 2."
    payloads = strategy.execute(text, context)

    assert len(payloads) == 2
    assert payloads[0].content == "Para 1.\nStill Para 1."
    assert payloads[1].content == "Para 2."


def test_paragraph_chunking_handles_multiple_newlines():
    strategy = ParagraphChunking()
    context = ExecutionContext()

    text = "Para 1.\n\n\n\nPara 2."
    payloads = strategy.execute(text, context)

    assert len(payloads) == 2
    assert payloads[0].content == "Para 1."
    assert payloads[1].content == "Para 2."


def test_paragraph_chunking_keeps_single_newlines_intact():
    strategy = ParagraphChunking()
    context = ExecutionContext()

    list_text = "Item 1\nItem 2\nItem 3"
    payloads = strategy.execute(list_text, context)

    assert len(payloads) == 1
    assert payloads[0].content == "Item 1\nItem 2\nItem 3"


def test_paragraph_chunking_structure_preservation():
    strategy = ParagraphChunking()
    context = ExecutionContext()

    payloads = strategy.execute(COMPLEX_MARKDOWN, context)

    assert len(payloads) == 8

    assert payloads[0].content == "# System Architecture"
    assert (
        payloads[1].content
        == "The system is built using a **microservices** architecture."
    )

    assert payloads[3].content.startswith("1. **API Gateway**")
    assert "Supports JWT." in payloads[3].content

    assert payloads[5].content.startswith("```json")
    assert '"source": "Client"' in payloads[5].content

    assert payloads[7].content == "> Warning: This module is deprecated."
    
def test_sentence_chunking_basic():
    strategy = SentenceChunking()
    context = ExecutionContext()
    text = "Hello world. This is a test! Is it working? Yes."
    results = strategy.execute(text, context)

    assert len(results) == 4
    assert results[0].content == "Hello world."
    assert results[1].content == "This is a test!"
    assert results[2].content == "Is it working?"
    assert results[3].content == "Yes."

def test_pipeline_paragraph_to_sentence_structure():
    spy = SpyStrategy()
    pipeline = Step(ParagraphChunking()) >> Step(SentenceChunking()) >> Step(spy)
    
    text = "Para 1 Sentence 1. Para 1 Sentence 2.\n\nPara 2 Sentence 1."
    pipeline.run(text)
    
    assert len(spy.received_chunks) == 3
    assert "Para 1 Sentence 1." in spy.received_chunks
    assert "Para 1 Sentence 2." in spy.received_chunks
    assert "Para 2 Sentence 1." in spy.received_chunks

def test_paragraph_annotated_with_sentences():
    strategy = ParagraphChunking()
    context = ExecutionContext(annotators={"sentences": SentenceChunking()})
    text = "Block A. Block B.\n\nBlock C."
    
    results = strategy.execute(text, context)
    
    assert len(results) == 2
    assert results[0].content == "Block A. Block B."
    
    assert "sentences" in results[0].annotations
    sentences_list = results[0].annotations["sentences"]
    assert len(sentences_list) == 2
    assert sentences_list[0].content == "Block A."
    assert sentences_list[1].content == "Block B."

def test_sentence_chunking_on_complex_markdown():
    strategy = SentenceChunking()
    context = ExecutionContext()
    
    payloads = strategy.execute(COMPLEX_MARKDOWN, context)
    
    assert len(payloads) > 1
    
    contents = [p.content for p in payloads]
    
    assert any("Supports OAuth2." in c for c in contents)
    assert any("This module is deprecated." in c for c in contents)
    
def test_sliding_window_logic():
    strategy = SlidingWindowChunking(window_size=3, overlap=1)
    context = ExecutionContext()
    text = "one two three four five"
    
    results = strategy.execute(text, context)
    
    assert len(results) == 2
    assert results[0].content == "one two three"
    assert results[1].content == "three four five"

def test_sliding_window_cleaning():
    strategy = SlidingWindowChunking(window_size=3, overlap=0)
    context = ExecutionContext()
    
    text = "word1   word2 word3" 
    
    results = strategy.execute(text, context)
    
    assert len(results) == 1
    assert results[0].content == "word1 word2 word3"

def test_sliding_window_validation():
    try:
        SlidingWindowChunking(window_size=5, overlap=5)
        assert False, "Should raise ValueError for overlap >= window_size"
    except ValueError:
        pass
    
def test_sliding_window_on_complex_markdown():
    """
    Verifies SlidingWindowChunking behavior on the shared complex markdown sample.
    Checks that the text is traversed and windowed correctly across markdown syntax.
    """
    strategy = SlidingWindowChunking(window_size=15, overlap=5)
    context = ExecutionContext()
    
    payloads = strategy.execute(COMPLEX_MARKDOWN, context)
    
    assert len(payloads) >= 3
    
    first_chunk = payloads[0].content
    assert "System Architecture" in first_chunk
    assert "microservices" in first_chunk
    
    last_chunk = payloads[-1].content
    assert "deprecated" in last_chunk
    
    combined_content = " ".join([p.content for p in payloads])
    assert "OAuth2" in combined_content
    assert "API Gateway" in combined_content
