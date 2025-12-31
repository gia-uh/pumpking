import pytest
from pumpking.models import ChunkPayload
from pumpking.protocols import ExecutionContext
from pumpking.strategies.basic import AdaptiveChunking

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

def test_adaptive_chunking_merges_short_sentences():
    """
    Verifies that the strategy successfully aggregates multiple short sentences into a single chunk.

    Adaptive Chunking aims to create semantically cohesive blocks that fit within a
    specific size range. This test provides a series of short sentences that are individually
    smaller than the 'min_chunk_size'.

    It asserts that the strategy buffers these sentences and emits them as a single
    consolidated ChunkPayload once the minimum threshold is met, rather than emitting
    three separate, tiny chunks.
    """
    strategy = AdaptiveChunking(min_chunk_size=25, max_chunk_size=100)
    context = ExecutionContext()
    text = "Short one. Short two. Short three."

    results = strategy.execute(text, context)

    assert len(results) == 1
    assert results[0].content == "Short one. Short two. Short three."

def test_adaptive_chunking_respects_max_limit():
    """
    Ensures that the strategy respects the upper bound of the chunk size.

    This test checks the overflow logic. It provides two sentences which, if combined,
    would exceed the 'max_chunk_size'. The strategy is expected to detect this potential
    overflow before adding the second sentence to the buffer.

    Consequently, it should emit the first sentence as a chunk (even if it might be
    slightly below optimal size, though here it satisfies min size logic if applicable)
    and start a new chunk for the second sentence.
    """
    strategy = AdaptiveChunking(min_chunk_size=10, max_chunk_size=20)
    context = ExecutionContext()
    text = "First sentence is long. Second sentence is also long."

    results = strategy.execute(text, context)

    assert len(results) == 2
    assert results[0].content == "First sentence is long."
    assert results[1].content == "Second sentence is also long."

def test_adaptive_chunking_on_complex_markdown():
    """
    Tests the adaptive logic on a structured document with varied content lengths.

    This test verifies that the strategy maintains its aggregation behavior even when
    processing complex content like Markdown headers and list items. It checks that
    key phrases from the beginning and end of the document are present in the output,
    ensuring that the split-and-merge process covers the entire text without dropping data.

    It also implicitly validates that the regex pattern used for splitting is robust enough
    to handle Markdown syntax as sentence boundaries where appropriate.
    """
    strategy = AdaptiveChunking(min_chunk_size=50, max_chunk_size=200)
    context = ExecutionContext()

    results = strategy.execute(COMPLEX_MARKDOWN, context)

    assert len(results) >= 1

    first_chunk = results[0].content
    assert "# System Architecture" in first_chunk
    assert "microservices" in first_chunk

    combined_content = "".join([p.content for p in results])
    assert "OAuth2" in combined_content
    assert "Warning: This module is deprecated." in combined_content

def test_adaptive_chunking_validation_error():
    """
    Verifies that invalid size configurations raise appropriate exceptions.

    The Adaptive Chunking strategy requires a logical range where 'min_chunk_size' is
    less than or equal to 'max_chunk_size'. It also requires positive integers.
    This test ensures that initializing the strategy with a minimum size larger than
    the maximum size triggers a ValueError, preventing logical errors during execution.
    """
    with pytest.raises(ValueError):
        AdaptiveChunking(min_chunk_size=100, max_chunk_size=50)