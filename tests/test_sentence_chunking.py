import pytest
from typing import List
from pumpking.models import ChunkPayload
from pumpking.pipeline import Step, PumpkingPipeline
from pumpking.protocols import ExecutionContext
from pumpking.strategies.base import BaseStrategy
from pumpking.strategies.basic import ParagraphChunking, SentenceChunking

# --- Mocks and Data ---

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
    """
    A simple strategy used to capture output for verification.
    """
    SUPPORTED_INPUTS = [str]
    PRODUCED_OUTPUT = ChunkPayload

    def __init__(self):
        self.received_payloads: List[ChunkPayload] = []

    def execute(self, data: str, context: ExecutionContext) -> ChunkPayload:
        payload = self._apply_annotators_to_payload(data, context)
        self.received_payloads.append(payload)
        return payload

# --- Tests ---

def test_sentence_chunking_basic():
    """
    Verifies the fundamental logic of splitting text into sentences based on punctuation.

    This test checks if the lookbehind regex pattern correctly identifies standard
    sentence terminators (periods, exclamation marks, question marks) followed by whitespace.
    It ensures that the resulting payloads contain distinct, complete sentences
    without losing the punctuation marks that define them.
    """
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
    pipeline.run(text)

    captured_chunks = [p.content for p in spy.received_payloads]

    assert len(captured_chunks) == 3
    assert "Para 1 Sentence 1." in captured_chunks
    assert "Para 1 Sentence 2." in captured_chunks
    assert "Para 2 Sentence 1." in captured_chunks

def test_paragraph_annotated_with_sentences():
    """
    Verifies that SentenceChunking functions correctly when used as an Annotator.
    
    In the Pumpking architecture, any Strategy can act as an Annotator. This test
    injects SentenceChunking into the ExecutionContext of a ParagraphChunking step.
    
    It asserts that:
    1. The main strategy (Paragraph) produces the primary content.
    2. The injected annotator (Sentence) is executed against that content.
    3. The resulting annotation is attached to the payload under the specified alias ("sentences").
    4. The annotation result is structurally correct (a list of sentence payloads).
    """
    strategy = ParagraphChunking()
    context = ExecutionContext()
    
    context.annotators = {"sentences": SentenceChunking()}
    
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
    """
    Tests the robustness of sentence segmentation when applied to mixed-content documents.

    Markdown documents often contain structures that mimic sentence boundaries or
    contain punctuation in non-grammatical contexts (like code blocks or headers).
    This test verifies that the strategy can extract meaningful sentence-like segments
    even from a noisy environment containing headers, lists, and code.

    It specifically checks for the presence of known sentences buried within list items
    and blockquotes to ensure the regex pattern is flexible enough to handle
    real-world formatting.
    """
    strategy = SentenceChunking()
    context = ExecutionContext()

    payloads = strategy.execute(COMPLEX_MARKDOWN, context)

    assert len(payloads) > 1

    contents = [p.content for p in payloads]

    assert any("Supports OAuth2." in c for c in contents)
    assert any("This module is deprecated." in c for c in contents)