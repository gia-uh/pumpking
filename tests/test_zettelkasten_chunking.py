import uuid
import pytest
from typing import List, Any
from unittest.mock import MagicMock, Mock

from pumpking.models import (
    ChunkPayload,
    ZettelChunkPayload,
    EntityChunkPayload
)
from pumpking.protocols import ExecutionContext, ZettelProviderProtocol
from pumpking.strategies.advanced import ZettelkastenChunking
from pumpking.strategies.base import BaseStrategy


class MockZettelProvider:
    def extract_zettels(self, chunks: List[ChunkPayload], **kwargs: Any) -> List[ZettelChunkPayload]:
        return []


class MockSplitter(BaseStrategy):
    def __init__(self, return_payloads: List[ChunkPayload]):
        self.return_payloads = return_payloads

    def execute(self, data: Any, context: ExecutionContext) -> List[ChunkPayload]:
        return self.return_payloads


def test_zettel_payload_structure():
    """
    Verifies that the ZettelChunkPayload correctly initializes with required fields,
    generates a default UUID, and supports the specific metadata fields for
    tags and graph relationships.
    """
    zettel_id = uuid.uuid4()
    related_id = uuid.uuid4()
    
    payload = ZettelChunkPayload(
        id=zettel_id,
        hypothesis="A synthetic test hypothesis.",
        tags=["Tag A", "Tag B"],
        related_zettel_ids=[related_id],
        content=None,
        children=[]
    )

    assert payload.id == zettel_id
    assert payload.hypothesis == "A synthetic test hypothesis."
    assert payload.tags == ["Tag A", "Tag B"]
    assert payload.related_zettel_ids == [related_id]
    assert payload.annotations == {}


def test_zettelkasten_strategy_execution_flow():
    """
    Tests the standard execution flow: 
    1. Input data is passed to the Splitter.
    2. Splitter results are converted/normalized.
    3. Normalized chunks are passed to the Provider.
    4. Provider results are returned by the Strategy.
    """
    input_text = "Some raw text."
    physical_chunk = ChunkPayload(content="Paragraph 1", id=uuid.uuid4())
    
    expected_zettel = ZettelChunkPayload(
        hypothesis="Hypothesis 1",
        children=[physical_chunk]
    )

    mock_splitter = MockSplitter(return_payloads=[physical_chunk])
    mock_provider = Mock(spec=ZettelProviderProtocol)
    mock_provider.extract_zettels.return_value = [expected_zettel]

    strategy = ZettelkastenChunking(splitter=mock_splitter, provider=mock_provider)
    context = ExecutionContext()

    results = strategy.execute(input_text, context)

    assert len(results) == 1
    assert results[0] == expected_zettel
    assert results[0].children[0] == physical_chunk
    
    mock_provider.extract_zettels.assert_called_once()
    call_args = mock_provider.extract_zettels.call_args[0][0]
    assert len(call_args) == 1
    assert call_args[0].content == "Paragraph 1"


def test_zettelkasten_strategy_annotation_logic():
    """
    Verifies the annotation lifecycle for Zettelkasten:
    1. Annotators run only on original evidence (splitter level).
    2. Hypothesis remains a pure metadata field without annotations.
    """
    physical_chunk = ChunkPayload(content="Original text about Apple.")
    zettel_result = ZettelChunkPayload(
        hypothesis="Apple Inc. released a new product.",
        children=[physical_chunk]
    )
    
    mock_splitter = MockSplitter(return_payloads=[physical_chunk])
    mock_provider = Mock(spec=ZettelProviderProtocol)
    mock_provider.extract_zettels.return_value = [zettel_result]
    
    mock_annotator = Mock(spec=BaseStrategy)
    
    strategy = ZettelkastenChunking(splitter=mock_splitter, provider=mock_provider)
    context = ExecutionContext(annotators={"ner": mock_annotator})
    
    results = strategy.execute("dummy input", context)
    
    annotated_zettel = results[0]
    
    assert "ner" not in annotated_zettel.annotations
    assert len(annotated_zettel.annotations) == 0
    
    for call in mock_annotator.execute.call_args_list:
        assert call[0][0] != "Apple Inc. released a new product."

def test_zettelkasten_strategy_handles_string_splitter_output():
    """
    Ensures robustness: If the injected splitter returns raw strings instead of 
    ChunkPayloads, the strategy should normalize them into ChunkPayloads 
    before sending them to the Provider.
    """
    mock_splitter = Mock(spec=BaseStrategy)
    mock_splitter.execute.return_value = ["String chunk 1", "String chunk 2"]

    mock_provider = Mock(spec=ZettelProviderProtocol)
    mock_provider.extract_zettels.return_value = []

    strategy = ZettelkastenChunking(splitter=mock_splitter, provider=mock_provider)
    context = ExecutionContext()

    strategy.execute("dummy input", context)

    mock_provider.extract_zettels.assert_called_once()
    passed_chunks = mock_provider.extract_zettels.call_args[0][0]
    
    assert len(passed_chunks) == 2
    assert isinstance(passed_chunks[0], ChunkPayload)
    assert passed_chunks[0].content == "String chunk 1"
    assert passed_chunks[1].content == "String chunk 2"


def test_zettelkasten_strategy_empty_input():
    """
    Verifies that the strategy handles empty input or empty splitter output 
    gracefully without invoking the provider.
    """
    mock_splitter = MockSplitter(return_payloads=[])
    mock_provider = Mock(spec=ZettelProviderProtocol)

    strategy = ZettelkastenChunking(splitter=mock_splitter, provider=mock_provider)
    context = ExecutionContext()

    results = strategy.execute("", context)

    assert results == []
    mock_provider.extract_zettels.assert_not_called()