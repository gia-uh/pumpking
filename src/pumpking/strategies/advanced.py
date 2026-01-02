import markdown
from html.parser import HTMLParser
from typing import List, Optional, Any, Union, Type, Dict
from pumpking.models import (
    ChunkPayload,
    EntityChunkPayload,
    TopicChunkPayload,
    ContextualChunkPayload,
    ZettelChunkPayload
)
from pumpking.protocols import (
    ExecutionContext,
    NERProviderProtocol,
    SummaryProviderProtocol,
    TopicProviderProtocol,
    ContextualProviderProtocol,
    ZettelProviderProtocol
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
            self.stack[-1].title = clean_text(self.stack[-1].title)

    def handle_data(self, data: str) -> None:
        if self.in_header:
            self.stack[-1].title += data
        else:
            self.stack[-1].content_buffer.append(data)

class HierarchicalChunking(BaseStrategy):
    """
    Parses Markdown structure and applies sub-strategies to section content.
    Produces a tree of ChunkPayloads where sections contain subsections as children.
    """

    SUPPORTED_INPUTS = [str, ChunkPayload]
    PRODUCED_OUTPUT = List[ChunkPayload]

    def __init__(self, strategies: List[BaseStrategy]) -> None:
        self.strategies = strategies

    def execute(self, data: Union[str, ChunkPayload], context: ExecutionContext) -> List[ChunkPayload]:
        if not data:
            return []

        text_content = data.content if isinstance(data, ChunkPayload) else str(data)

        if not text_content:
            return []

        html_content = markdown.markdown(text_content)
        parser = _MarkdownStructureParser()
        parser.feed(html_content)

        top_level_payloads = []

        root_text = parser.root.get_text_content()
        
        if root_text.strip() and self.strategies:
            root_body_chunks = self._apply_strategy_chain(root_text, context)
            top_level_payloads.extend(root_body_chunks)

        for child_node in parser.root.children:
            top_level_payloads.append(self._create_section_payload(child_node, context))

        return top_level_payloads

    def _create_section_payload(
        self, node: _SectionNode, context: ExecutionContext
    ) -> ChunkPayload:
        body_payloads = []
        text_content = node.get_text_content()

        if text_content.strip() and self.strategies:
            body_payloads = self._apply_strategy_chain(text_content, context)

        subsection_payloads = []
        for child_node in node.children:
            subsection_payloads.append(
                self._create_section_payload(child_node, context)
            )

        header_clean = node.title
        full_content = f"{header_clean}\n{text_content}".strip()
        full_content_raw = f"{'#' * node.level} {node.title}\n{text_content}"

        payload = self._apply_annotators_to_payload(
            content=full_content, context=context, content_raw=full_content_raw
        )

        all_children = body_payloads + subsection_payloads
        if all_children:
            payload.children = all_children

        return payload

    def _apply_strategy_chain(
        self, content: str, context: ExecutionContext
    ) -> List[ChunkPayload]:
        if not self.strategies:
            return []

        primary_strategy = self.strategies[0]
        return primary_strategy.execute(content, context)

class EntityBasedChunking(BaseStrategy):
    """
    Groups text segments based on Named Entity Recognition (NER) analysis.
    """
    SUPPORTED_INPUTS = [str, ChunkPayload]
    PRODUCED_OUTPUT = List[EntityChunkPayload]

    def __init__(
        self,
        ner_provider: Union[NERProviderProtocol, Type[NERProviderProtocol]] = LLMProvider,
        splitter: Union[BaseStrategy, Type[BaseStrategy]] = SentenceChunking,
        window_size: int = 20,
        window_overlap: int = 5,
    ) -> None:
        if isinstance(ner_provider, type):
            self.ner_provider = ner_provider()
        else:
            self.ner_provider = ner_provider

        if isinstance(splitter, type):
            self.splitter = splitter()
        else:
            self.splitter = splitter

        self.window_size = window_size
        self.window_overlap = window_overlap

    def execute(self, data: Union[str, ChunkPayload], context: ExecutionContext) -> List[EntityChunkPayload]:
        if not data:
            return []

        base_chunks = self.splitter.execute(data, context)
        if not base_chunks:
            return []

        entity_payloads = self.ner_provider.extract_entities(
            base_chunks,
            window_size=self.window_size,
            window_overlap=self.window_overlap
        )

        if context.annotators and entity_payloads:
            sub_ctx = ExecutionContext()
            for payload in entity_payloads:
                for alias, strategy in context.annotators.items():
                    try:
                        payload.annotations[alias] = strategy.execute(payload.entity, sub_ctx)
                    except Exception as e:
                        payload.annotations[alias] = {"error": str(e)}

        return entity_payloads

class SummaryChunking(BaseStrategy):
    """
    Implements a semantic compression strategy using AI-generated summaries.
    """
    SUPPORTED_INPUTS: List[Any] = [str, ChunkPayload]
    PRODUCED_OUTPUT: Any = List[ChunkPayload]

    def __init__(
        self,
        provider: Union[SummaryProviderProtocol, Type[SummaryProviderProtocol]] = LLMProvider,
        splitter: Union[BaseStrategy, Type[BaseStrategy]] = AdaptiveChunking,
        **kwargs: Any,
    ) -> None:
        if isinstance(provider, type):
            self.provider = provider()
        else:
            self.provider = provider

        if isinstance(splitter, type):
            self.splitter = splitter()
        else:
            self.splitter = splitter

        self.provider_kwargs = kwargs

    def execute(self, data: Union[str, ChunkPayload], context: ExecutionContext) -> List[ChunkPayload]:
        base_chunks = self.splitter.execute(data, context)
        
        if not base_chunks:
            return []

        payloads_to_process = []
        for item in base_chunks:
            if isinstance(item, ChunkPayload):
                payloads_to_process.append(item)
            else:
                payloads_to_process.append(ChunkPayload(content=str(item)))

        summary_payloads = self.provider.summarize(payloads_to_process, **self.provider_kwargs)

        if context.annotators and summary_payloads:
            sub_ctx = ExecutionContext()
            for payload in summary_payloads:
                if payload.content:
                    for alias, strategy in context.annotators.items():
                        try:
                            # Annotate the summary text
                            res = strategy.execute(payload.content, sub_ctx)
                            payload.annotations[alias] = res
                        except Exception as e:
                            payload.annotations[alias] = {"error": str(e)}

        return summary_payloads
class TopicBasedChunking(BaseStrategy):
    """
    Organizes text content into thematic clusters based on semantic topic assignment.
    """
    SUPPORTED_INPUTS = [str, ChunkPayload]
    PRODUCED_OUTPUT = List[TopicChunkPayload]

    def __init__(
        self,
        topic_provider: Union[TopicProviderProtocol, Type[TopicProviderProtocol]] = LLMProvider,
        splitter: Union[BaseStrategy, Type[BaseStrategy]] = ParagraphChunking,
        **kwargs: Any,
    ) -> None:
        if isinstance(topic_provider, type):
            self.topic_provider = topic_provider()
        else:
            self.topic_provider = topic_provider

        if isinstance(splitter, type):
            self.splitter = splitter()
        else:
            self.splitter = splitter

        self.provider_kwargs = kwargs

    def execute(self, data: Union[str, ChunkPayload], context: ExecutionContext) -> List[TopicChunkPayload]:
        if not data:
            return []

        base_chunks = self.splitter.execute(data, context)
        if not base_chunks:
            return []

        payloads_to_process = []
        for item in base_chunks:
            if isinstance(item, ChunkPayload):
                payloads_to_process.append(item)
            else:
                payloads_to_process.append(ChunkPayload(content=str(item)))

        topic_payloads = self.topic_provider.assign_topics(payloads_to_process, **self.provider_kwargs)

        if context.annotators and topic_payloads:
            sub_ctx = ExecutionContext()
            for payload in topic_payloads:
                for alias, strategy in context.annotators.items():
                    try:
                        # Annotate the topic label
                        payload.annotations[alias] = strategy.execute(payload.topic, sub_ctx)
                    except Exception as e:
                        payload.annotations[alias] = {"error": str(e)}
        
        return topic_payloads

class ContextualChunking(BaseStrategy):
    """
    Orchestrates the enrichment of text chunks with situational context.

    This strategy operates on an ordered sequence of text. It delegates the
    segmentation to a splitter strategy and then passes the ordered list of
    fragments to a Contextual Provider.

    The strategy assumes that the context for any given chunk is derivable
    from the collection of chunks provided. It coordinates the 1-to-1 mapping
    between the original chunks and the context strings returned by the provider,
    producing specialized ContextualChunkPayloads.
    """

    SUPPORTED_INPUTS: List[Any] = [str]
    PRODUCED_OUTPUT: Any = List[ContextualChunkPayload]

    def __init__(
        self,
        provider: Union[ContextualProviderProtocol, Type[ContextualProviderProtocol]] = LLMProvider,
        splitter: Union[BaseStrategy, Type[BaseStrategy]] = AdaptiveChunking,
        **kwargs: Any,
    ) -> None:
        if isinstance(provider, type):
            self.provider = provider()
        else:
            self.provider = provider

        if isinstance(splitter, type):
            self.splitter = splitter()
        else:
            self.splitter = splitter

        self.provider_kwargs = kwargs

    def execute(self, data: Union[str, ChunkPayload], context: ExecutionContext) -> List[ContextualChunkPayload]:
        base_chunks = self.splitter.execute(data, context)
        
        if not base_chunks:
            return []

        payloads_to_process = []
        for item in base_chunks:
            if isinstance(item, ChunkPayload):
                payloads_to_process.append(item)
            else:
                payloads_to_process.append(ChunkPayload(content=str(item)))

        contextual_payloads = self.provider.assign_context(payloads_to_process, **self.provider_kwargs)

        if context.annotators and contextual_payloads:
            sub_ctx = ExecutionContext()
            for payload in contextual_payloads:
                if payload.context:
                    for alias, strategy in context.annotators.items():
                        try:
                            res = strategy.execute(payload.context, sub_ctx)
                            payload.annotations[alias] = res
                        except Exception as e:
                            payload.annotations[alias] = {"error": str(e)}

        return contextual_payloads
class ZettelkastenChunking(BaseStrategy):
    """
    Implements the Zettelkasten method for knowledge extraction.
    """
    SUPPORTED_INPUTS = [str, ChunkPayload]
    PRODUCED_OUTPUT = List[ZettelChunkPayload]

    def __init__(
        self, 
        provider: Union[ZettelProviderProtocol, Type[ZettelProviderProtocol]] = LLMProvider,
        splitter: Union[BaseStrategy, Type[BaseStrategy]] = ParagraphChunking,
        **kwargs: Any
    ):
        if isinstance(provider, type):
            self.provider = provider()
        else:
            self.provider = provider

        if isinstance(splitter, type):
            self.splitter = splitter()
        else:
            self.splitter = splitter

        self.provider_kwargs = kwargs

    def execute(self, data: Union[str, ChunkPayload], context: ExecutionContext) -> List[ZettelChunkPayload]:
        base_chunks = self.splitter.execute(data, context)
        if not base_chunks:
            return []

        payloads_to_process = []
        for item in base_chunks:
            if isinstance(item, ChunkPayload):
                payloads_to_process.append(item)
            else:
                payloads_to_process.append(ChunkPayload(content=str(item)))

        zettels = self.provider.extract_zettels(payloads_to_process, **self.provider_kwargs)

        if context.annotators and zettels:
            sub_ctx = ExecutionContext()
            for payload in zettels:
                if payload.hypothesis:
                    for alias, strategy in context.annotators.items():
                        try:
                            payload.annotations[alias] = strategy.execute(payload.hypothesis, sub_ctx)
                        except Exception as e:
                            payload.annotations[alias] = {"error": str(e)}

        return zettels