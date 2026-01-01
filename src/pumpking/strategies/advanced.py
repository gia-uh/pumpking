import markdown
from html.parser import HTMLParser
from typing import List, Optional, Any, Union, Type, Dict
from pumpking.models import (
    ChunkPayload,
    ChunkNode,
    EntityChunkPayload,
    EntityChunkNode,
    TopicChunkNode,
    TopicChunkPayload,
    ContextualChunkNode,
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

    SUPPORTED_INPUTS = [str]
    PRODUCED_OUTPUT = List[ChunkPayload]

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

    This strategy operates by first fragmenting the text into atomic units (typically sentences)
    and ensuring these units are fully annotated by the pipeline's active annotators.
    It then employs a Named Entity Recognition provider to identify entities within these
    segments and maps the entities back to the specific segments where they appear.

    A key architectural feature of this strategy is the preservation of object references.
    If a single text segment (e.g., a sentence) contains references to multiple entities,
    the same segment payload object is included in the children list of multiple
    entity payloads. This ensures that expensive annotations computed during the splitting
    phase are not duplicated, and the memory footprint remains efficient.

    Attributes:
        ner_provider (NERProviderProtocol): The service instance used to extract entities.
        splitter (BaseStrategy): The strategy instance used to split text before extraction.
        window_size (int): The number of segments processed together in a single batch.
        window_overlap (int): The number of overlapping segments between batches.
    """

    SUPPORTED_INPUTS = [str]
    PRODUCED_OUTPUT = List[EntityChunkPayload]

    def __init__(
        self,
        ner_provider: Union[NERProviderProtocol, Type[NERProviderProtocol]] = LLMProvider,
        splitter: Union[BaseStrategy, Type[BaseStrategy]] = SentenceChunking,
        window_size: int = 20,
        window_overlap: int = 5,
    ) -> None:
        """
        Initializes the strategy.

        Args:
            ner_provider: The NER provider to use. Can be an initialized instance OR
                          a class type (e.g., LLMProvider) which will be instantiated
                          with default settings. Defaults to LLMProvider class.
            splitter: The splitting strategy. Can be an instance OR a class type.
                      Defaults to SentenceChunking class.
        """
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

    def execute(self, data: str, context: ExecutionContext) -> List[ChunkPayload]:
        if not data:
            return []

        sentence_payloads = self.splitter.execute(data, context)

        if not sentence_payloads:
            return []

        clean_segments_for_ner = [p.content for p in sentence_payloads]

        ner_results = self.ner_provider.extract_entities(
            clean_segments_for_ner,
            window_size=self.window_size,
            window_overlap=self.window_overlap,
        )

        entity_payloads = []

        for result in ner_results:
            children_references = []

            for index in result.indices:
                if 0 <= index < len(sentence_payloads):
                    children_references.append(sentence_payloads[index])

            if children_references:
                entity_payload = EntityChunkPayload(
                    content=result.entity,
                    content_raw="",  
                    entity=result.entity,
                    type=result.label,
                    children=children_references,
                )
                entity_payloads.append(entity_payload)

        return entity_payloads


class SummaryChunking(BaseStrategy):
    """
    Implements a semantic compression strategy that replaces text blocks with their
    AI-generated summaries.

    This strategy functions as a transformation layer in the pipeline. It delegates
    the initial segmentation of the text to a configurable 'splitter' strategy
    (defaulting to AdaptiveChunking). Once the text is segmented, it iterates
    through each block and employs a 'provider' to generate a concise summary.

    The resulting ChunkPayload objects are constructed such that the 'content'
    field holds the generated summary. This ensures that downstream tasks, such
    as embedding generation or sentiment analysis, operate on the distilled
    information. The original text of the block is preserved in the 'content_raw'
    field for lineage, auditing, or retrieval purposes.

    Attributes:
        provider (SummaryProviderProtocol): The component responsible for summarizing text.
        splitter (BaseStrategy): The strategy used to segment the input data before summarization.
        kwargs (dict): Configuration parameters forwarded to the provider during execution.
    """

    SUPPORTED_INPUTS: List[Any] = [str]
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

    def execute(self, data: str, context: ExecutionContext) -> List[ChunkPayload]:
        if not data:
            return []

        raw_blocks = self.splitter.execute(data, ExecutionContext())

        summary_payloads = []

        for block in raw_blocks:
            if not block.content:
                continue

            summary_text = self.provider.summarize(block.content, **self.provider_kwargs)

            payload = self._apply_annotators_to_payload(
                content=summary_text,
                context=context,
                content_raw=block.content 
            )
            
            summary_payloads.append(payload)

        return summary_payloads


class TopicBasedChunking(BaseStrategy):
    """
    Organizes text content into thematic clusters based on semantic topic assignment.

    This strategy operates on the principle of semantic reorganization rather than strict linear
    segmentation. It first decomposes the input text into granular units (typically paragraphs)
    and ensures these units are processed by the pipeline's annotators. It then leverages a
    Topic Provider (e.g., an LLM) to classify each unit into one or more relevant topics.

    A defining characteristic of this strategy is its support for multi-membership: a single
    paragraph can be relevant to multiple topics. In such cases, the strategy maps the
    paragraph to multiple topic containers. Crucially, it uses Python object references
    to ensure that the underlying paragraph payload (and its expensive annotations) is
    stored in memory only once, even if it appears in the children lists of several
    different TopicChunkPayloads.

    Attributes:
        topic_provider (TopicProviderProtocol): The service responsible for analyzing text
            segments and returning a list of applicable topics for each. Defaults to LLMProvider.
        splitter (BaseStrategy): The strategy used to create the atomic units of text
            to be classified. Defaults to ParagraphChunking.
        provider_kwargs (dict): Additional configuration arguments passed dynamically
            to the topic provider during execution (e.g., taxonomy generation settings).
    """

    SUPPORTED_INPUTS = [str]
    PRODUCED_OUTPUT = List[TopicChunkPayload]

    def __init__(
        self,
        topic_provider: Union[TopicProviderProtocol, Type[TopicProviderProtocol]] = LLMProvider,
        splitter: Union[BaseStrategy, Type[BaseStrategy]] = ParagraphChunking,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the strategy with a provider and a splitter.

        Args:
            topic_provider: The provider implementation for topic assignment. Accepts either
                            an initialized instance or a class type. If a class type is provided,
                            it will be instantiated with default parameters.
            splitter: The strategy used to split the source text. Accepts an instance or a class type.
            **kwargs: Arbitrary keyword arguments forwarded to the topic_provider's
                      assign_topics method (e.g., batch_size, model, taxonomy_mode).
        """
        if isinstance(topic_provider, type):
            self.topic_provider = topic_provider()
        else:
            self.topic_provider = topic_provider

        if isinstance(splitter, type):
            self.splitter = splitter()
        else:
            self.splitter = splitter

        self.provider_kwargs = kwargs

    def execute(self, data: str, context: ExecutionContext) -> List[ChunkPayload]:
        if not data:
            return []

        base_chunks = self.splitter.execute(data, context)

        if not base_chunks:
            return []

        clean_texts = [c.content for c in base_chunks]

        topics_matrix = self.topic_provider.assign_topics(clean_texts, **self.provider_kwargs)

        if len(topics_matrix) != len(base_chunks):
            return []

        topic_map: Dict[str, List[ChunkPayload]] = {}

        for chunk_reference, assigned_topics in zip(base_chunks, topics_matrix):
            for topic in assigned_topics:
                if topic not in topic_map:
                    topic_map[topic] = []
                topic_map[topic].append(chunk_reference)

        results = []
        for topic_label, grouped_chunks in topic_map.items():
            topic_payload = TopicChunkPayload(
                content=topic_label,
                content_raw="", 
                topic=topic_label,
                children=grouped_chunks
            )
            results.append(topic_payload)

        return results


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

    def execute(self, data: str, context: ExecutionContext) -> List[ContextualChunkPayload]:
        if not data:
            return []

        base_chunks = self.splitter.execute(data, ExecutionContext())
        
        if not base_chunks:
            return []

        chunk_texts = [b.content for b in base_chunks if b.content]
        
        context_strings = self.provider.assign_context(chunk_texts, **self.provider_kwargs)

        if len(context_strings) != len(chunk_texts):
            context_strings = context_strings[:len(chunk_texts)] + [""] * (len(chunk_texts) - len(context_strings))

        results = []
        
        for base_chunk, ctx_str in zip(base_chunks, context_strings):
            if not base_chunk.content:
                continue

            standard_payload = self._apply_annotators_to_payload(
                content=base_chunk.content,
                context=context,
                content_raw=base_chunk.content_raw or base_chunk.content
            )

            payload = ContextualChunkPayload(
                context=ctx_str,
                **standard_payload.model_dump()
            )
            
            results.append(payload)

        return results

class ZettelkastenChunking(BaseStrategy):
    """
    Implements the Zettelkasten method for knowledge extraction. This strategy 
    transforms linear documents into a network of atomic, interconnected notes.

    The process involves three main stages:
    1. Splitting: The input document is segmented into physical units (e.g., 
       paragraphs) using a subordinate splitting strategy.
    2. Extraction: A ZettelProvider analyzes the segments to produce atomic 
       ZettelChunkPayloads, resolving relationships and assigning evidence.
    3. Annotation: If annotators are present in the execution context, they 
       are applied to the 'hypothesis' of each Zettel, enriching the node 
       with additional metadata (e.g., Named Entities).
    """

    SUPPORTED_INPUTS = [str]
    PRODUCED_OUTPUT = List[ZettelChunkPayload]

    def __init__(
        self, 
        provider: Union[ZettelProviderProtocol, Type[ZettelProviderProtocol]] = LLMProvider,
        splitter: Union[BaseStrategy, Type[BaseStrategy]] = ParagraphChunking,
        **kwargs: Any
    ):
        """
        Initializes the strategy with a physical splitter and a semantic provider.

        Args:
            provider: The provider implementation for concept extraction. Accepts 
                      either an initialized instance or a class type. Defaults to 
                      LLMProvider.
            splitter: The strategy used to split the source text into physical 
                      fragments. Accepts an instance or a class type. Defaults to 
                      ParagraphChunking.
            **kwargs: Arbitrary keyword arguments forwarded to the provider's 
                      extract_zettels method.
        """
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
        """
        Executes the Zettelkasten extraction pipeline.

        Args:
            data: The input content, either as a raw string or a ChunkPayload.
            context: The execution context containing configuration and 
                registered annotators.

        Returns:
            A list of ZettelChunkPayload objects representing the extracted 
            knowledge graph.
        """
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

        if context.annotators:
            annotation_context = ExecutionContext()
            
            for alias, strategy in context.annotators.items():
                for zettel in zettels:
                    if zettel.hypothesis:
                        annotation_result = strategy.execute(zettel.hypothesis, annotation_context)
                        zettel.annotations[alias] = annotation_result

        return zettels