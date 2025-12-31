import pytest
from typing import List
from pumpking.models import ChunkPayload
from pumpking.pipeline import Step, PumpkingPipeline
from pumpking.protocols import ExecutionContext
from pumpking.strategies.base import BaseStrategy
from pumpking.strategies.basic import RegexChunking

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

def test_regex_chunking_strategy_pure_logic():
    """
    Verifies the core splitting logic of RegexChunking in isolation.
    
    This test ensures that:
    1. The strategy correctly splits text based on the newline pattern.
    2. It returns a list of ChunkPayload objects.
    3. The content of the payloads matches the expected markdown sections.
    """
    strategy = RegexChunking(pattern=r"\n\n+")
    context = ExecutionContext()
    
    payloads = strategy.execute(COMPLEX_MARKDOWN, context)

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
    """
    Verifies RegexChunking behavior when orchestrated within a Pipeline.
    
    This test ensures that:
    1. The Step adapter correctly handles the list of Payloads produced by RegexChunking.
    2. The downstream step (SpyStrategy) receives the individual segments as input.
    3. The document graph (root_node) is constructed correctly.
    """
    chunker = RegexChunking(pattern=r"\n\n+")
    spy = SpyStrategy()

    pipeline = PumpkingPipeline(Step(chunker) >> Step(spy))

    root_node = pipeline.run(COMPLEX_MARKDOWN)

    assert root_node.document == COMPLEX_MARKDOWN

    captured_contents = [p.content for p in spy.received_payloads]
    
    assert len(captured_contents) >= 6
    assert "# System Architecture" in captured_contents
    assert "## Core Components" in captured_contents

    assert captured_contents[0] == "# System Architecture"
    assert "microservices" in captured_contents[1]