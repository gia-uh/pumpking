import pytest
from typing import List
from pumpking.models import ChunkPayload
from pumpking.protocols import ExecutionContext
from pumpking.strategies.base import BaseStrategy
from pumpking.strategies.basic import ParagraphChunking
from pumpking.strategies.advanced import HierarchicalChunking

# --- Tests ---

def test_hierarchical_chunking_structure_h1_to_body():
    """
    Verifies that the strategy correctly parses a top-level Markdown header and associates its body text.
    """
    strategies = [ParagraphChunking()]
    strategy = HierarchicalChunking(strategies=strategies)
    context = ExecutionContext()

    text = "# Section A\nIntro text."
    results = strategy.execute(text, context)

    assert len(results) == 1
    root_payload = results[0]

    assert "Section A" in root_payload.content
    
    assert root_payload.children is not None
    assert len(root_payload.children) == 1
    
    child_body = root_payload.children[0]
    assert "Intro text" in child_body.content

def test_hierarchical_chunking_deep_nesting():
    """
    Validates recursive parsing of nested Markdown headers (H1 -> H2).
    """
    strategies = [ParagraphChunking()]
    strategy = HierarchicalChunking(strategies=strategies)
    context = ExecutionContext()

    text = "# Main\n\n## Sub\nDetails."
    results = strategy.execute(text, context)

    assert len(results) == 1
    h1_node = results[0]
    
    assert "Main" in h1_node.content
    assert len(h1_node.children) == 1 

    h2_node = h1_node.children[0]
    assert "Sub" in h2_node.content
    assert len(h2_node.children) == 1
    
    body_node = h2_node.children[0]
    assert "Details" in body_node.content

def test_hierarchical_chunking_preamble_handling():
    """
    Verifies behavior when text exists before the first header.
    """
    strategies = [ParagraphChunking()]
    strategy = HierarchicalChunking(strategies=strategies)
    context = ExecutionContext()

    text = "Preamble text.\n# Chapter 1\nContent."
    results = strategy.execute(text, context)

    assert len(results) == 2
    
    assert "Preamble text" in results[0].content
    assert results[0].children is None or len(results[0].children) == 0
    
    assert "Chapter 1" in results[1].content
    assert len(results[1].children) == 1

def test_hierarchical_chunking_mixed_content_and_subsections():
    """
    Tests a complex scenario where a section contains both immediate body text and subsections.
    """
    strategies = [ParagraphChunking()]
    strategy = HierarchicalChunking(strategies=strategies)
    context = ExecutionContext()

    text = "# Root\nImmediate Intro.\n## Branch\nLeaf content."
    results = strategy.execute(text, context)

    root = results[0]
    assert len(root.children) == 2
    
    child_1 = root.children[0]
    assert "Immediate Intro" in child_1.content
    
    child_2 = root.children[1]
    assert "Branch" in child_2.content
    assert len(child_2.children) == 1

def test_hierarchical_chunking_normalizes_markdown():
    """
    Ensures that the `content_raw` field contains a valid Markdown representation.
    
    Note: Since the strategy parses via HTML and reconstructs the structure, 
    excess whitespace in headers (e.g., '##   Title') is normalized to standard 
    Markdown ('## Title'). This test verifies that valid reconstruction occurs.
    """
    strategies = [ParagraphChunking()]
    strategy = HierarchicalChunking(strategies=strategies)
    context = ExecutionContext()

    text = "##   Spaced Header\nBody."
    results = strategy.execute(text, context)

    h2_payload = results[0]
    
    # The strategy normalizes multiple spaces to a single space
    assert "## Spaced Header" in h2_payload.content_raw
    assert "Spaced Header" in h2_payload.content 

def test_hierarchical_chunking_empty_input():
    """
    Verifies that the strategy handles empty input gracefully without errors.
    """
    strategies = [ParagraphChunking()]
    strategy = HierarchicalChunking(strategies=strategies)
    context = ExecutionContext()

    results = strategy.execute("", context)
    assert results == []