import pytest
from typing import List, Any
from pumpking.models import ChunkPayload, EntityChunkPayload
from pumpking.protocols import ExecutionContext, NERProviderProtocol
from pumpking.strategies.base import BaseStrategy
from pumpking.strategies.advanced import EntityBasedChunking

# --- Mocks for Testing ---


class MockAnnotator(BaseStrategy):
    """
    Spy annotator to verify that annotations are computed only once
    and preserved across shared references.
    """

    def __init__(self):
        self.call_count = 0

    def execute(self, data: str, context: ExecutionContext) -> dict:
        self.call_count += 1
        return {"spy_checked": True, "id": self.call_count}


class MockNERProvider(NERProviderProtocol):
    """
    Simulates the LLMProvider behavior.
    Crucial Change: It now returns a list of fully formed EntityChunkPayloads,
    matching the updated protocol, instead of intermediate dictionaries.
    """

    def __init__(self, fixed_results: List[EntityChunkPayload]):
        self.fixed_results = fixed_results

    def extract_entities(
        self, chunks: List[ChunkPayload], **kwargs: Any
    ) -> List[EntityChunkPayload]:
        return self.fixed_results


class MockSplitter(BaseStrategy):
    """
    Simulates SentenceChunking.
    Splits text by periods for simplicity in testing.
    """

    def execute(self, data: str, context: ExecutionContext) -> List[ChunkPayload]:
        segments = [s.strip() for s in data.split(".") if s.strip()]
        payloads = []
        for seg in segments:
            p = ChunkPayload(content=seg, content_raw=seg)
            p = self._apply_annotators_to_payload(p.content, context, p.content_raw)
            payloads.append(p)
        return payloads


# --- Tests ---


def test_entity_chunking_groups_correctly():
    """
    Verifies that the strategy correctly passes through the payloads
    returned by the provider.
    """
    chunk1 = ChunkPayload(
        content="Elon Musk runs SpaceX", content_raw="Elon Musk runs SpaceX"
    )
    chunk2 = ChunkPayload(content="He runs Tesla", content_raw="He runs Tesla")

    mock_payloads = [
        EntityChunkPayload(
            entity="Elon Musk",
            type="PER",
            children=[chunk1, chunk2],
            content="Elon Musk",
        ),
        EntityChunkPayload(
            entity="SpaceX", type="ORG", children=[chunk1], content="SpaceX"
        ),
    ]

    provider = MockNERProvider(mock_payloads)
    splitter = MockSplitter()

    strategy = EntityBasedChunking(ner_provider=provider, splitter=splitter)
    context = ExecutionContext()

    text = "Elon Musk runs SpaceX. He runs Tesla."

    results = strategy.execute(text, context)

    assert len(results) == 2

    elon_node = results[0]
    assert elon_node.entity == "Elon Musk"
    assert elon_node.type == "PER"
    assert len(elon_node.children) == 2
    assert elon_node.children[0].content == "Elon Musk runs SpaceX"

    spacex_node = results[1]
    assert spacex_node.entity == "SpaceX"
    assert len(spacex_node.children) == 1


def test_entity_chunking_annotators_run_once():
    """
    Verifies that annotators are executed during the splitting phase
    and NOT re-executed during the grouping phase.
    """
    spy_annotator = MockAnnotator()
    context = ExecutionContext()
    context.annotators = {"spy": spy_annotator}

    provider = MockNERProvider([])
    splitter = MockSplitter()

    strategy = EntityBasedChunking(ner_provider=provider, splitter=splitter)

    text = "Single Sentence."

    strategy.execute(text, context)

    assert spy_annotator.call_count == 1


def test_entity_chunking_empty_input():
    """
    Verifies graceful handling of empty strings.
    """
    strategy = EntityBasedChunking(
        ner_provider=MockNERProvider([]), splitter=MockSplitter()
    )
    context = ExecutionContext()

    assert strategy.execute("", context) == []
    assert strategy.execute(None, context) == []
