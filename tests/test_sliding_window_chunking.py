import pytest
from pumpking.models import ChunkPayload
from pumpking.protocols import ExecutionContext
from pumpking.strategies.basic import SlidingWindowChunking

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

def test_sliding_window_logic():
    """
    Verifies the core sliding window algorithm on a simple word sequence.

    This test checks the mathematical correctness of the windowing logic.
    Given a sequence of words, a defined window size, and an overlap count:
    1. It ensures the first chunk contains exactly 'window_size' words.
    2. It ensures the second chunk starts 'overlap' words before the end of the first chunk.
    3. It verifies that the step size (stride) is correctly calculated as (window_size - overlap).

    Input: "one two three four five", Window: 3, Overlap: 1
    Expected Step: 2
    Chunk 1: "one two three" (Indices 0, 1, 2)
    Chunk 2: "three four five" (Indices 2, 3, 4)
    """
    strategy = SlidingWindowChunking(window_size=3, overlap=1)
    context = ExecutionContext()
    text = "one two three four five"

    results = strategy.execute(text, context)

    assert len(results) == 2
    assert results[0].content == "one two three"
    assert results[1].content == "three four five"

def test_sliding_window_cleaning():
    """
    Ensures that the strategy normalizes whitespace within the generated windows.

    The strategy operates by splitting the source text into a list of words
    (which naturally consumes whitespace) and then joining them back together
    with single spaces. This test confirms that irregular spacing in the source
    input (e.g., multiple spaces between words) is collapsed into clean,
    normalized text in the output payloads.
    """
    strategy = SlidingWindowChunking(window_size=3, overlap=0)
    context = ExecutionContext()

    text = "word1   word2 word3"

    results = strategy.execute(text, context)

    assert len(results) == 1
    assert results[0].content == "word1 word2 word3"

def test_sliding_window_validation():
    """
    Verifies that the strategy enforces valid configuration constraints.

    The sliding window algorithm requires logical parameters to function:
    1. Window size must be strictly positive (cannot have a window of 0 words).
    2. Overlap must be non-negative.
    3. Overlap must be strictly smaller than the window size (otherwise the loop would never advance).

    This test asserts that invalid combinations raise the appropriate ValueError.
    """
    with pytest.raises(ValueError):
        SlidingWindowChunking(window_size=5, overlap=5)
    
    with pytest.raises(ValueError):
        SlidingWindowChunking(window_size=0, overlap=0)

def test_sliding_window_on_complex_markdown():
    """
    Tests the strategy's behavior on structured content (Markdown).

    Sliding Window is a mechanical strategy that ignores semantic boundaries like
    paragraphs or headers, treating the text purely as a stream of words.
    This test verifies that:
    1. The content is preserved and searchable within the chunks.
    2. The sliding mechanism traverses the entire document.
    3. Keywords from the beginning and end of the document are successfully captured.
    
    This confirms that while structure might be broken (e.g., a header might satisfy
    the word count mid-sentence), the data ingestion remains complete.
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