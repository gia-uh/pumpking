import pytest

from typing import List, Any
from pumpking.models import ChunkPayload
from pumpking.pipeline import annotate, Step, PumpkingPipeline
from pumpking.protocols import ExecutionContext
from pumpking.strategies.base import BaseStrategy
from pumpking.strategies.basic import (
    RegexChunking,
    FixedSizeChunking,
    ParagraphChunking,
    SentenceChunking,
    SlidingWindowChunking,
    AdaptiveChunking,
)
from pumpking.models import NERResult, TopicChunkNode, TopicChunkPayload
from pumpking.protocols import NERProviderProtocol, SummaryProviderProtocol
from pumpking.strategies.advanced import (
    EntityBasedChunking,
    HierarchicalChunking,
    SummaryChunkingStrategy,
    TopicBasedChunking,
)
from pumpking.strategies.providers import LLMProvider


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

    assert root_node.document == COMPLEX_MARKDOWN

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

    assert payloads[1].content == "e The system is bui"

    assert payloads[1].content_raw == "e\n\nThe system is bui"


def test_fixed_size_chunking_overlap_consistency():
    chunk_size = 50
    overlap = 10
    strategy = FixedSizeChunking(chunk_size=chunk_size, overlap=overlap)
    context = ExecutionContext()

    payloads = strategy.execute(COMPLEX_MARKDOWN, context)

    first_chunk_end = payloads[0].content_raw[-overlap:]
    second_chunk_start = payloads[1].content_raw[:overlap]

    assert first_chunk_end == second_chunk_start


def test_fixed_size_chunking_preserves_total_content():
    chunk_size = 100
    strategy = FixedSizeChunking(chunk_size=chunk_size, overlap=0)
    context = ExecutionContext()

    payloads = strategy.execute(COMPLEX_MARKDOWN, context)

    reconstructed = "".join([p.content_raw for p in payloads])
    assert reconstructed == COMPLEX_MARKDOWN


def test_fixed_size_chunking_validation_error():
    with pytest.raises(ValueError):
        FixedSizeChunking(chunk_size=0)

    with pytest.raises(ValueError):
        FixedSizeChunking(chunk_size=5, overlap=6)


def test_paragraph_chunking_splits_on_double_newlines():
    strategy = ParagraphChunking()
    context = ExecutionContext()

    text = "Para 1.\nStill Para 1.\n\nPara 2."
    payloads = strategy.execute(text, context)

    assert len(payloads) == 2
    assert payloads[0].content == "Para 1.\nStill Para 1."
    assert payloads[1].content == "Para 2."


def test_paragraph_chunking_handles_multiple_newlines():
    strategy = ParagraphChunking()
    context = ExecutionContext()

    text = "Para 1.\n\n\n\nPara 2."
    payloads = strategy.execute(text, context)

    assert len(payloads) == 2
    assert payloads[0].content == "Para 1."
    assert payloads[1].content == "Para 2."


def test_paragraph_chunking_keeps_single_newlines_intact():
    strategy = ParagraphChunking()
    context = ExecutionContext()

    list_text = "Item 1\nItem 2\nItem 3"
    payloads = strategy.execute(list_text, context)

    assert len(payloads) == 1
    assert payloads[0].content == "Item 1\nItem 2\nItem 3"


def test_paragraph_chunking_structure_preservation():
    strategy = ParagraphChunking()
    context = ExecutionContext()

    payloads = strategy.execute(COMPLEX_MARKDOWN, context)

    assert len(payloads) == 8

    assert payloads[0].content == "# System Architecture"
    assert (
        payloads[1].content
        == "The system is built using a **microservices** architecture."
    )

    assert payloads[3].content.startswith("1. **API Gateway**")
    assert "Supports JWT." in payloads[3].content

    assert payloads[5].content.startswith("```json")
    assert '"source": "Client"' in payloads[5].content

    assert payloads[7].content == "> Warning: This module is deprecated."


def test_sentence_chunking_basic():
    strategy = SentenceChunking()
    context = ExecutionContext()
    text = "Hello world. This is a test! Is it working? Yes."
    results = strategy.execute(text, context)

    assert len(results) == 4
    assert results[0].content == "Hello world."
    assert results[1].content == "This is a test!"
    assert results[2].content == "Is it working?"
    assert results[3].content == "Yes."


def test_pipeline_paragraph_to_sentence_structure():
    spy = SpyStrategy()
    pipeline = Step(ParagraphChunking()) >> Step(SentenceChunking()) >> Step(spy)

    text = "Para 1 Sentence 1. Para 1 Sentence 2.\n\nPara 2 Sentence 1."
    pipeline.run(text)

    assert len(spy.received_chunks) == 3
    assert "Para 1 Sentence 1." in spy.received_chunks
    assert "Para 1 Sentence 2." in spy.received_chunks
    assert "Para 2 Sentence 1." in spy.received_chunks


def test_paragraph_annotated_with_sentences():
    strategy = ParagraphChunking()
    context = ExecutionContext(annotators={"sentences": SentenceChunking()})
    text = "Block A. Block B.\n\nBlock C."

    results = strategy.execute(text, context)

    assert len(results) == 2
    assert results[0].content == "Block A. Block B."

    assert "sentences" in results[0].annotations
    sentences_list = results[0].annotations["sentences"]
    assert len(sentences_list) == 2
    assert sentences_list[0].content == "Block A."
    assert sentences_list[1].content == "Block B."


def test_sentence_chunking_on_complex_markdown():
    strategy = SentenceChunking()
    context = ExecutionContext()

    payloads = strategy.execute(COMPLEX_MARKDOWN, context)

    assert len(payloads) > 1

    contents = [p.content for p in payloads]

    assert any("Supports OAuth2." in c for c in contents)
    assert any("This module is deprecated." in c for c in contents)


def test_sliding_window_logic():
    strategy = SlidingWindowChunking(window_size=3, overlap=1)
    context = ExecutionContext()
    text = "one two three four five"

    results = strategy.execute(text, context)

    assert len(results) == 2
    assert results[0].content == "one two three"
    assert results[1].content == "three four five"


def test_sliding_window_cleaning():
    strategy = SlidingWindowChunking(window_size=3, overlap=0)
    context = ExecutionContext()

    text = "word1   word2 word3"

    results = strategy.execute(text, context)

    assert len(results) == 1
    assert results[0].content == "word1 word2 word3"


def test_sliding_window_validation():
    try:
        SlidingWindowChunking(window_size=5, overlap=5)
        assert False, "Should raise ValueError for overlap >= window_size"
    except ValueError:
        pass


def test_sliding_window_on_complex_markdown():
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


def test_adaptive_chunking_merges_short_sentences():
    strategy = AdaptiveChunking(min_chunk_size=25, max_chunk_size=100)
    context = ExecutionContext()
    text = "Short one. Short two. Short three."

    results = strategy.execute(text, context)

    assert len(results) == 1
    assert results[0].content == "Short one. Short two. Short three."


def test_adaptive_chunking_respects_max_limit():
    strategy = AdaptiveChunking(min_chunk_size=10, max_chunk_size=20)
    context = ExecutionContext()
    text = "First sentence is long. Second sentence is also long."

    results = strategy.execute(text, context)

    assert len(results) == 2
    assert results[0].content == "First sentence is long."
    assert results[1].content == "Second sentence is also long."


def test_adaptive_chunking_on_complex_markdown():
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
    try:
        AdaptiveChunking(min_chunk_size=100, max_chunk_size=50)
        assert False
    except ValueError:
        pass


def test_hierarchical_chunking_content_aggregation():
    strategies = [ParagraphChunking()]
    strategy = HierarchicalChunking(strategies=strategies)
    context = ExecutionContext()

    text = "# Header 1\nBody line 1.\n## Header 2\nBody line 2."
    results = strategy.execute(text, context)

    assert len(results) == 1
    h1_node = results[0]

    assert "Header 1" in h1_node.content
    assert "Body line 1" in h1_node.content
    assert "Header 2" in h1_node.content

    assert "# Header 1" in h1_node.content_raw
    assert "## Header 2" in h1_node.content_raw


def test_hierarchical_chunking_structure_and_children():
    strategies = [ParagraphChunking()]
    strategy = HierarchicalChunking(strategies=strategies)
    context = ExecutionContext()

    text = "# Main\nIntro.\n## Sub\nDetails."
    results = strategy.execute(text, context)

    main_node = results[0]
    assert len(main_node.children) == 2

    para_node = main_node.children[0]
    assert "Intro." in para_node.content

    sub_node = main_node.children[1]
    assert "Sub" in sub_node.content
    assert "Details." in sub_node.content

    assert len(sub_node.children) == 1
    assert "Details." in sub_node.children[0].content


def test_hierarchical_chunking_deep_chain_raw():
    strategies = [ParagraphChunking(), SentenceChunking()]
    strategy = HierarchicalChunking(strategies=strategies)
    context = ExecutionContext()

    text = "# A\nPara. Sent."
    results = strategy.execute(text, context)

    node_a = results[0]
    assert "# A" in node_a.content_raw
    assert "Para. Sent." in node_a.content_raw

    para_node = node_a.children[0]
    assert "Para. Sent." == para_node.content

    sent_node = para_node.children[0]
    assert "Para." in sent_node.content


class MockNERProvider(NERProviderProtocol):
    def __init__(self, predefined_results: List[NERResult] = None):
        self.predefined_results = predefined_results or []

    # Updated to accept **kwargs (window_size, etc)
    def extract_entities(self, sentences: List[str], **kwargs: Any) -> List[NERResult]:
        if not self.predefined_results:
            if not sentences:
                return []
            return [
                NERResult(
                    entity="Default Mock",
                    label="MISC",
                    indices=list(range(len(sentences))),
                )
            ]
        return self.predefined_results


def test_entity_chunking_complex_narrative_overlap_and_coref():
    text = "Apple released the Vision Pro. It implies a new era. Tim Cook announced it in Cupertino. The device costs $3500."

    mock_results = [
        NERResult(entity="Apple", label="ORG", indices=[0, 2]),
        NERResult(entity="Vision Pro", label="PROD", indices=[0, 1, 2, 3]),
        NERResult(entity="Tim Cook", label="PER", indices=[2]),
    ]

    provider = MockNERProvider(predefined_results=mock_results)
    strategy = EntityBasedChunking(ner_provider=provider)
    context = ExecutionContext()

    payloads = strategy.execute(text, context)

    assert len(payloads) == 3

    apple_node = next(p for p in payloads if p.entity == "Apple")
    assert apple_node.children is not None
    assert len(apple_node.children) == 2
    assert "Apple released" in apple_node.children[0].content
    assert "Tim Cook announced" in apple_node.children[1].content


def test_entity_chunking_preserves_messy_format_in_raw():
    raw_text = "**SpaceX** launched the _Starship_ containing [Link](http://x.com)."

    mock_results = [NERResult(entity="SpaceX", label="ORG", indices=[0])]

    provider = MockNERProvider(predefined_results=mock_results)
    strategy = EntityBasedChunking(ner_provider=provider)
    context = ExecutionContext()

    payloads = strategy.execute(raw_text, context)

    assert len(payloads) == 1
    child = payloads[0].children[0]

    assert "**" not in child.content
    assert "http" not in child.content
    assert "[" not in child.content

    assert child.content_raw == raw_text


def test_entity_chunking_discontinuous_segments():
    text = "Python is great. Java is verbose. Python is also dynamic."

    mock_results = [
        NERResult(entity="Python", label="LANG", indices=[0, 2]),
        NERResult(entity="Java", label="LANG", indices=[1]),
    ]

    provider = MockNERProvider(predefined_results=mock_results)
    strategy = EntityBasedChunking(ner_provider=provider)
    payloads = strategy.execute(text, ExecutionContext())

    python_node = next(p for p in payloads if p.entity == "Python")

    assert len(python_node.children) == 2
    assert "is great" in python_node.children[0].content
    assert "is also dynamic" in python_node.children[1].content
    assert "Java" not in "".join([c.content for c in python_node.children])


class MockSummaryProvider(SummaryProviderProtocol):
    """
    Mock provider that performs a simple deterministic transformation.
    """

    def summarize(self, text: str, **kwargs: Any) -> str:
        return f"summarized_content_of_[{text}]"


def test_summary_chunking_basic_flow():
    mock_provider = MockSummaryProvider()
    strategy = SummaryChunkingStrategy(
        provider=mock_provider, min_chunk_size=10, max_chunk_size=100
    )
    context = ExecutionContext()

    text = "Original Text"

    payloads = strategy.execute(text, context)

    assert len(payloads) == 1
    assert payloads[0].content == "summarized_content_of_[Original Text]"
    assert payloads[0].content_raw == "Original Text"


def test_summary_chunking_integration_with_adaptive_logic():
    mock_provider = MockSummaryProvider()
    strategy = SummaryChunkingStrategy(
        provider=mock_provider, min_chunk_size=5, max_chunk_size=20
    )
    context = ExecutionContext()

    text = "Short. Another short. A very long sentence that exceeds limit."

    payloads = strategy.execute(text, context)

    assert len(payloads) >= 2

    for p in payloads:
        expected_summary = f"summarized_content_of_[{p.content_raw}]"
        assert p.content == expected_summary
        assert p.content_raw is not None


def test_summary_chunking_defaults_to_llm_provider():
    strategy = SummaryChunkingStrategy()
    assert isinstance(strategy.provider, LLMProvider)


class MockTopicProvider:
    """
    Mock provider for thematic classification.
    """
    def assign_topics(self, chunks: List[str], **kwargs: Any) -> List[List[str]]:
        """
        Simulates global taxonomy assignment based on segment content.
        """
        if not chunks:
            return []
            
        results = []
        for text in chunks:
            topics = []
            if "tecnología" in text.lower() or "blockchain" in text.lower():
                topics.append("Tecnología")
            if "economía" in text.lower() or "pib" in text.lower():
                topics.append("Economía")
            if not topics:
                topics.append("General")
            results.append(topics)
        return results

def test_topic_based_chunking_logic():
    """
    Tests the core grouping logic and multi-topic support.
    """
    provider = MockTopicProvider()
    strategy = TopicBasedChunking(topic_provider=provider)
    context = ExecutionContext()
    
    data = "Blockchain y tecnología.\n\nEl PIB y la economía.\n\nTecnología financiera."
    
    payloads = strategy.execute(data, context)
    
    assert len(payloads) >= 2
    tech_payload = next(p for p in payloads if p.topic == "Tecnología")
    assert len(tech_payload.children) == 2
    
    econ_payload = next(p for p in payloads if p.topic == "Economía")
    assert len(econ_payload.children) == 1

def test_topic_chunking_pipeline_integration():
    """
    Tests integration with PumpkingPipeline using Step objects.
    """
    provider = MockTopicProvider()
    strategy = TopicBasedChunking(topic_provider=provider)
    
    topic_step = Step(strategy, alias="AnalisisTematico")
    pipeline = PumpkingPipeline() >> topic_step
    
    doc_root = pipeline.run("Blockchain en la economía.\n\nContenido general.")
    
    assert len(doc_root.children[0].children) > 0
    
    topic_nodes = [
        node for node in doc_root.children[0].children 
        if isinstance(node, TopicChunkNode)
    ]
    
    assert len(topic_nodes) > 0
    for node in topic_nodes:
        assert node.strategy_label == "AnalisisTematico"
        assert hasattr(node, "topic")
        assert node.content is None

def test_topic_chunking_empty_input():
    """
    Verifies behavior with empty strings.
    """
    provider = MockTopicProvider()
    strategy = TopicBasedChunking(topic_provider=provider)
    
    payloads = strategy.execute("", ExecutionContext())
    assert payloads == []

def test_topic_chunking_to_node_conversion():
    """
    Verifies polymorphic conversion from payload to node.
    """
    provider = MockTopicProvider()
    strategy = TopicBasedChunking(topic_provider=provider)
    
    payload = TopicChunkPayload(
        topic="Salud",
        children=[{"content": "Texto de prueba", "content_raw": "Texto de prueba"}]
    )
    
    node = strategy.to_node(payload)
    
    assert isinstance(node, TopicChunkNode)
    assert node.topic == "Salud"
    assert len(node.children) == 1