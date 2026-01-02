import pytest
from typing import List, Union
from pumpking.strategies.basic import RegexChunking
from pumpking.models import ChunkPayload
from pumpking.protocols import ExecutionContext
from pumpking.pipeline import Step, PumpkingPipeline
from pumpking.strategies.base import BaseStrategy

# --- Fixtures and Mocks ---

COMPLEX_MARKDOWN = """# System Architecture

The system is composed of three main modules:
1. Core Kernel
2. Network Interface
3. Storage Manager

## Core Kernel
Handles process scheduling and memory management.

## Network Interface
Manages TCP/IP stack.
"""


class SpyStrategy(BaseStrategy):
    """
    Mock strategy to inspect what data is passed to the next step.
    UPDATED: Handles input normalization (str vs Payload) to support chaining.
    """

    def execute(self, data: Union[str, ChunkPayload], context: ExecutionContext) -> ChunkPayload:
        content = data.content if isinstance(data, ChunkPayload) else str(data)
        
        payload = self._apply_annotators_to_payload(content, context)
        return payload


# --- Tests ---


def test_regex_chunking_logic_only():
    """
    Verifies the core logic of RegexChunking in isolation.
    """
    chunker = RegexChunking(pattern=r"\n\n+")
    context = ExecutionContext()
    
    results = chunker.execute(COMPLEX_MARKDOWN, context)
    
    assert len(results) == 4
    
    assert "# System Architecture" in results[0].content
    assert "The system is composed" in results[1].content
    assert "## Core Kernel" in results[2].content
    assert "## Network Interface" in results[3].content
    
    assert all(isinstance(r, ChunkPayload) for r in results)


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
    
    chunker_node_branches = root_node.branches
    assert len(chunker_node_branches) == 4
    
    for branch in chunker_node_branches:
        assert len(branch.branches) == 1
        spy_node = branch.branches[0]
        assert spy_node.strategy_label == "SpyStrategy"
        
        assert spy_node.results[0].content == branch.results[0].content