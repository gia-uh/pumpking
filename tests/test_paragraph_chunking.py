import pytest
from typing import List
from pumpking.models import ChunkPayload
from pumpking.protocols import ExecutionContext
from pumpking.strategies.basic import ParagraphChunking, SentenceChunking

# --- Data Constants ---

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

# --- Tests ---

def test_paragraph_chunking_splits_on_double_newlines():
    """
    Verifies that the strategy correctly identifies paragraph boundaries marked by double newlines.

    This test asserts that a standard double newline sequence serves as a delimiter,
    splitting the text into distinct payloads. It confirms that the content within
    the paragraphs remains intact while the delimiters themselves are handled
    by the splitting logic.
    """
    strategy = ParagraphChunking()
    context = ExecutionContext()

    text = "Para 1.\nStill Para 1.\n\nPara 2."
    payloads = strategy.execute(text, context)

    assert len(payloads) == 2
    assert payloads[0].content == "Para 1.\nStill Para 1."
    assert payloads[1].content == "Para 2."

def test_paragraph_chunking_handles_multiple_newlines():
    """
    Ensures that excessive vertical whitespace does not create empty chunks.

    This test checks the robustness of the regex pattern (specifically the '+' quantifier).
    It validates that multiple consecutive newlines (e.g., three or four) are treated
    as a single split point, preventing the generation of empty or "ghost" payloads
    between valid paragraphs.
    """
    strategy = ParagraphChunking()
    context = ExecutionContext()

    text = "Para 1.\n\n\n\nPara 2."
    payloads = strategy.execute(text, context)

    assert len(payloads) == 2
    assert payloads[0].content == "Para 1."
    assert payloads[1].content == "Para 2."

def test_paragraph_chunking_keeps_single_newlines_intact():
    """
    Verifies that single newlines are treated as content, not split points.

    This behavior is crucial for preserving the structure of lists, code blocks,
    or soft-wrapped text within a single logical paragraph. The test asserts that
    items separated by a single newline remain grouped within a single ChunkPayload.
    """
    strategy = ParagraphChunking()
    context = ExecutionContext()

    list_text = "Item 1\nItem 2\nItem 3"
    payloads = strategy.execute(list_text, context)

    assert len(payloads) == 1
    assert payloads[0].content == "Item 1\nItem 2\nItem 3"

def test_paragraph_chunking_structure_preservation():
    """
    Validates the strategy against a complex Markdown document structure.

    This comprehensive test checks how the paragraph chunking applies to a mix
    of headers, text blocks, lists, and code blocks. It ensures that semantic
    blocks separated by whitespace (like headers vs. content or code blocks vs. text)
    are correctly segmented into separate payloads.

    It asserts specific content matches to verify that Markdown syntax characters
    are preserved within their respective chunks.
    """
    strategy = ParagraphChunking()
    context = ExecutionContext()

    payloads = strategy.execute(COMPLEX_MARKDOWN, context)

    assert len(payloads) == 8

    assert payloads[0].content == "# System Architecture"
    assert payloads[1].content == "The system is built using a **microservices** architecture."

    assert payloads[3].content.startswith("1. **API Gateway**")
    assert "Supports JWT." in payloads[3].content

    assert payloads[5].content.startswith("```json")
    assert '"source": "Client"' in payloads[5].content

    assert payloads[7].content == "> Warning: This module is deprecated."

def test_paragraph_annotated_with_sentences():
    """
    Verifies the Inversion of Control (IoC) mechanism for annotations within this strategy.

    This test simulates a pipeline scenario where 'SentenceChunking' is injected
    as an annotator via the ExecutionContext. It asserts that:
    1. The ParagraphChunking strategy performs its primary split (paragraphs).
    2. It successfully delegates to the base class to apply the injected annotator.
    3. The resulting payloads contain the expected annotation metadata (nested sentence chunks).

    This proves that ParagraphChunking correctly participates in the annotation ecosystem.
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