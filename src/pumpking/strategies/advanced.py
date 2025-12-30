import markdown
from html.parser import HTMLParser
from typing import List, Optional, Any
from pumpking.models import (
    ChunkPayload,
    ChunkNode,
    EntityChunkPayload,
    EntityChunkNode,
    TopicChunkNode,
    TopicChunkPayload,
)
from pumpking.protocols import (
    ExecutionContext,
    NERProviderProtocol,
    SummaryProviderProtocol,
    TopicProviderProtocol,
)
from pumpking.strategies.base import BaseStrategy
from pumpking.strategies.basic import (
    SentenceChunking,
    AdaptiveChunking,
    ParagraphChunking,
)
from pumpking.strategies.providers import LLMProvider
from pumpking.utils import clean_text


class _SectionNode:
    def __init__(self, level: int, title: str = "") -> None:
        self.level = level
        self.title = title
        self.content_buffer: List[str] = []
        self.children: List["_SectionNode"] = []

    def get_text_content(self) -> str:
        return "".join(self.content_buffer)


class _MarkdownStructureParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.root = _SectionNode(level=0)
        self.stack: List[_SectionNode] = [self.root]
        self.in_header = False

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> None:
        if tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            level = int(tag[1])
            self.in_header = True

            while self.stack[-1].level >= level:
                self.stack.pop()

            new_node = _SectionNode(level=level)
            self.stack[-1].children.append(new_node)
            self.stack.append(new_node)

    def handle_endtag(self, tag: str) -> None:
        if tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            self.in_header = False
            current_node = self.stack[-1]
            current_node.title = clean_text(current_node.title)

    def handle_data(self, data: str) -> None:
        if not data:
            return

        current_node = self.stack[-1]

        if self.in_header:
            current_node.title += data
        else:
            current_node.content_buffer.append(data)


class HierarchicalChunking(BaseStrategy):
    """
    Parses Markdown structure and applies sub-strategies to section content.

    This strategy converts the input Markdown to HTML to reliably detect
    document structure (headers). It creates a tree of chunks representing
    sections and sub-sections. The text content of each section is processed
    by the provided list of sub-strategies. The resulting payload includes
    aggregated content and raw content for the entire subtree.
    """

    def __init__(self, strategies: List[BaseStrategy]) -> None:
        self.strategies = strategies

    def execute(self, data: str, context: ExecutionContext) -> List[ChunkPayload]:
        if not data:
            return []

        html_content = markdown.markdown(data)

        parser = _MarkdownStructureParser()
        parser.feed(html_content)

        top_level_payloads = []

        root_text = parser.root.get_text_content()
        if root_text and root_text.strip() and self.strategies:
            root_body_chunks = self._apply_strategy_chain(
                root_text, self.strategies, context
            )
            top_level_payloads.extend(root_body_chunks)

        for child_node in parser.root.children:
            top_level_payloads.append(self._create_section_payload(child_node, context))

        return top_level_payloads

    def _create_section_payload(
        self, node: _SectionNode, context: ExecutionContext
    ) -> ChunkPayload:
        body_payloads = []
        children_section_payloads = []

        text_content = node.get_text_content()
        if text_content and text_content.strip() and self.strategies:
            body_payloads = self._apply_strategy_chain(
                text_content, self.strategies, context
            )

        for child_node in node.children:
            children_section_payloads.append(
                self._create_section_payload(child_node, context)
            )

        header_clean = node.title
        header_raw = f"{'#' * node.level} {node.title}"

        body_clean = " ".join([p.content for p in body_payloads])
        body_raw = "".join([p.content_raw or p.content for p in body_payloads])

        children_clean = " ".join([p.content for p in children_section_payloads])
        children_raw = "\n".join(
            [p.content_raw or p.content for p in children_section_payloads]
        )

        full_content = f"{header_clean} {body_clean} {children_clean}".strip()

        parts_raw = [header_raw]
        if body_raw:
            parts_raw.append(body_raw)
        if children_raw:
            parts_raw.append(children_raw)
        full_content_raw = "\n".join(parts_raw)

        all_children_nodes = body_payloads + children_section_payloads

        payload = self._apply_annotators_to_payload(
            content=full_content, context=context, content_raw=full_content_raw
        )

        if all_children_nodes:
            payload.children = all_children_nodes

        return payload

    def _apply_strategy_chain(
        self, content: str, strategies: List[BaseStrategy], context: ExecutionContext
    ) -> List[ChunkPayload]:
        if not strategies:
            return []

        current_strategy = strategies[0]
        remaining_strategies = strategies[1:]

        chunks = current_strategy.execute(content, context)

        if remaining_strategies:
            for chunk in chunks:
                children = self._apply_strategy_chain(
                    chunk.content, remaining_strategies, context
                )
                if children:
                    chunk.children = children

        return chunks


class EntityBasedChunking(BaseStrategy):
    """
    Strategy that groups text by entities using a NER provider.
    Delegates splitting to SentenceChunking for robust sentence boundary detection.
    """

    def __init__(
        self,
        ner_provider: NERProviderProtocol = LLMProvider(),
        splitter: BaseStrategy = SentenceChunking(),
        window_size: int = 20,
        window_overlap: int = 5,
    ) -> None:
        self.ner_provider = ner_provider or LLMProvider()
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.splitter = SentenceChunking()

    def execute(self, data: str, context: ExecutionContext) -> List[ChunkPayload]:
        if not data:
            return []

        sentence_payloads = self.splitter.execute(data, context)

        if not sentence_payloads:
            return []

        for p in sentence_payloads:
            if p.content_raw:
                p.content = clean_text(p.content_raw, strip_markdown=True)

        clean_segments_for_ner = [p.content for p in sentence_payloads]

        ner_results = self.ner_provider.extract_entities(
            clean_segments_for_ner,
            window_size=self.window_size,
            window_overlap=self.window_overlap,
        )

        entity_payloads = []

        for result in ner_results:
            children = []

            for index in result.indices:
                if 0 <= index < len(sentence_payloads):
                    children.append(sentence_payloads[index])

            if children:
                entity_node = EntityChunkPayload(
                    entity=result.entity,
                    type=result.label,
                    content=None,
                    children=children,
                )
                entity_payloads.append(entity_node)

        return entity_payloads

    def to_node(self, payload: ChunkPayload) -> ChunkNode:
        if isinstance(payload, EntityChunkPayload):
            children_nodes = []
            if payload.children:
                children_nodes = [
                    ChunkNode(
                        content=chunk.content,
                        content_raw=chunk.content_raw,
                        annotations=chunk.annotations,
                        children=[],
                    )
                    for chunk in payload.children
                ]

            return EntityChunkNode(
                entity=payload.entity,
                type=payload.type,
                content=payload.content,
                content_raw=payload.content_raw,
                annotations=payload.annotations,
                children=children_nodes,
            )

        return ChunkNode(
            content=payload.content,
            content_raw=payload.content_raw,
            annotations=payload.annotations,
            children=[],
        )


class SummaryChunkingStrategy(BaseStrategy):
    """
    Splits text into adaptive chunks and generates a summary for each.

    This strategy uses AdaptiveChunking to create semantically complete text blocks
    within a specific size range. Each block is then summarized by the provided
    SummaryProvider.
    """
    def __init__(
        self, 
        provider: SummaryProviderProtocol = LLMProvider(), 
        min_chunk_size: int = 1000, 
        max_chunk_size: int = 4000
    ) -> None:
        self.provider = provider
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.splitter = AdaptiveChunking(
            min_chunk_size=self.min_chunk_size, 
            max_chunk_size=self.max_chunk_size
        )

    def execute(self, data: str, context: ExecutionContext) -> List[ChunkPayload]:
        """
        Executes adaptive splitting and summarizes each resulting block.
        """
        if not data:
            return []

        empty_context = ExecutionContext()
        raw_blocks = self.splitter.execute(data, empty_context)
        
        summary_payloads = []
        
        for block in raw_blocks:
            if not block.content:
                continue

            summary_text = self.provider.summarize(block.content)
            
            payload = self._apply_annotators_to_payload(
                content=summary_text,
                context=context,
                content_raw=block.content
            )
            summary_payloads.append(payload)
            
        return summary_payloads


class TopicBasedChunking(BaseStrategy):
    """
    Strategy that groups text fragments into thematic containers.

    Allows multi-membership where a single fragment can belong to several topics.
    """

    def __init__(
        self,
        topic_provider: TopicProviderProtocol,
        splitter: BaseStrategy = ParagraphChunking(),
        **kwargs: Any,
    ) -> None:
        self.topic_provider = topic_provider
        self.splitter = splitter
        self.provider_kwargs = kwargs

    def execute(self, data: str, context: ExecutionContext) -> List[TopicChunkPayload]:
        """
        Splits text using the internal splitter and groups results by topics.
        """
        base_chunks = self.splitter.execute(data, context)
        if not base_chunks:
            return []

        texts = [c.content for c in base_chunks if c.content]
        if not texts:
            return []

        topics_matrix = self.topic_provider.assign_topics(texts, **self.provider_kwargs)

        topic_map: Dict[str, List[ChunkPayload]] = {}
        for chunk, topics in zip(base_chunks, topics_matrix):
            for topic in topics:
                if topic not in topic_map:
                    topic_map[topic] = []
                topic_map[topic].append(chunk)

        return [TopicChunkPayload(topic=t, children=c) for t, c in topic_map.items()]

    def to_node(self, payload: Any) -> ChunkNode:
        """
        Converts a TopicChunkPayload into a TopicChunkNode preserving hierarchy.
        """
        if isinstance(payload, TopicChunkPayload):
            children_nodes = (
                [super(TopicBasedChunking, self).to_node(c) for c in payload.children]
                if payload.children
                else []
            )

            return TopicChunkNode(topic=payload.topic, children=children_nodes)

        return super().to_node(payload)
