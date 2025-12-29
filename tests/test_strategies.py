import pytest

from typing import List, Any
from pumpking.models import ChunkPayload
from pumpking.pipeline import Step
from pumpking.protocols import ExecutionContext
from pumpking.strategies.base import BaseStrategy
from pumpking.strategies.basic import RegexChunking, FixedSizeChunking

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
"""

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
    assert payloads[1].content == "e\n\nThe system is bui"

def test_fixed_size_chunking_overlap_consistency():
    chunk_size = 50
    overlap = 10
    strategy = FixedSizeChunking(chunk_size=chunk_size, overlap=overlap)
    context = ExecutionContext()
    
    payloads = strategy.execute(COMPLEX_MARKDOWN, context)
    
    first_chunk_end = payloads[0].content[-overlap:]
    second_chunk_start = payloads[1].content[:overlap]
    
    assert first_chunk_end == second_chunk_start

def test_fixed_size_chunking_preserves_total_content():
    chunk_size = 100
    strategy = FixedSizeChunking(chunk_size=chunk_size, overlap=0)
    context = ExecutionContext()
    
    payloads = strategy.execute(COMPLEX_MARKDOWN, context)
    
    reconstructed = "".join([p.content for p in payloads])
    assert reconstructed == COMPLEX_MARKDOWN

def test_fixed_size_chunking_validation_error():
    with pytest.raises(ValueError):
        FixedSizeChunking(chunk_size=0)
    
    with pytest.raises(ValueError):
        FixedSizeChunking(chunk_size=5, overlap=6)