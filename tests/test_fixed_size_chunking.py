import pytest
from pumpking.models import ChunkPayload
from pumpking.protocols import ExecutionContext
from pumpking.strategies.basic import FixedSizeChunking

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

def test_fixed_size_chunking_cuts_structure_blindly():
    """
    Verifies that the strategy strictly adheres to character limits, ignoring semantic boundaries.

    This test confirms the fundamental behavior of fixed-size chunking: it must split text
    exactly at the specified character count, even if that split occurs in the middle of
    a word or a Markdown structure. This behavior distinguishes it from semantic or
    delimiter-based strategies.

    The test sets a small chunk size (20 characters) and asserts that the resulting
    payloads contain fragmented words, proving that the strategy is operating mechanically
    rather than semantically.
    """
    chunk_size = 20
    strategy = FixedSizeChunking(chunk_size=chunk_size, overlap=0)
    context = ExecutionContext()

    payloads = strategy.execute(COMPLEX_MARKDOWN, context)

    assert len(payloads) > 0
    
    assert payloads[0].content == "# System Architectur"
    assert payloads[1].content == "e The system is bui"
    
    assert payloads[1].content_raw == "e\n\nThe system is bui"

def test_fixed_size_chunking_overlap_consistency():
    """
    Validates the data continuity provided by the overlap mechanism.

    This test ensures that when an overlap is configured, the end of one chunk is
    duplicated as the start of the subsequent chunk. This is critical for downstream
    tasks (like embedding retrieval) where context might be lost at the boundary
    of a split.

    By checking that the last 'N' characters of the first chunk match the first 'N'
    characters of the second chunk, we verify the sliding window logic is implemented
    correctly.
    """
    chunk_size = 50
    overlap = 10
    strategy = FixedSizeChunking(chunk_size=chunk_size, overlap=overlap)
    context = ExecutionContext()

    payloads = strategy.execute(COMPLEX_MARKDOWN, context)

    first_chunk_end = payloads[0].content_raw[-overlap:]
    second_chunk_start = payloads[1].content_raw[:overlap]

    assert first_chunk_end == second_chunk_start

def test_fixed_size_chunking_preserves_total_content():
    """
    Ensures that the splitting process is lossless when overlap is zero.

    This test reconstructs the original document by concatenating the raw content
    of all generated payloads. It verifies that no characters are dropped or added
    during the chunking process.

    This guarantee is essential for auditing purposes and ensures that the
    sum of the parts equals the whole, provided no overlap introduces duplication.
    """
    chunk_size = 100
    strategy = FixedSizeChunking(chunk_size=chunk_size, overlap=0)
    context = ExecutionContext()

    payloads = strategy.execute(COMPLEX_MARKDOWN, context)

    reconstructed = "".join([p.content_raw for p in payloads])
    assert reconstructed == COMPLEX_MARKDOWN

def test_fixed_size_chunking_validation_error():
    """
    Verifies that the strategy rejects invalid configuration parameters during initialization.

    This test checks for three invalid states:
    1. A chunk size of zero (which would result in infinite loops).
    2. A chunk size smaller than the overlap (which is logically impossible).
    3. Negative values for overlap.

    Ensuring these ValueError exceptions are raised protects the pipeline from
    hanging or producing corrupted data due to misconfiguration.
    """
    with pytest.raises(ValueError):
        FixedSizeChunking(chunk_size=0)

    with pytest.raises(ValueError):
        FixedSizeChunking(chunk_size=5, overlap=6)

    with pytest.raises(ValueError):
        FixedSizeChunking(chunk_size=10, overlap=-1)