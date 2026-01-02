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

        # Currently applies only the primary strategy to the text body
        primary_strategy = self.strategies[0]
        return primary_strategy.execute(content, context)

class EntityBasedChunking(BaseStrategy):
    """
    An advanced compound strategy that identifies and extracts named entities from text.

    This strategy orchestrates a workflow designed to classify specific segments of 
    text into predefined categories. It decomposes input content into atomic units 
    and leverages a specialized NER provider to extract entities.

    It adheres to the NERProviderProtocol and is designed to preserve the complex 
    lineage established by the provider, where an entity may be linked to multiple 
    source fragments.
    """

    def __init__(
        self, 
        ner_provider: Any, 
        splitter: Optional[Any] = None, 
        provider_kwargs: Optional[dict] = None,
        strip_markdown: bool = False,
        collapse_whitespace: bool = True,
        **extra_kwargs: Any
    ):
        """
        Initializes the entity strategy, supporting both instances and class types.

        Args:
            ner_provider: An instance or class conforming to NERProviderProtocol.
            splitter: An optional instance or class for content decomposition.
            provider_kwargs: Configuration for the NER provider.
            strip_markdown: Flag to remove markdown during text cleaning.
            collapse_whitespace: Flag to unify whitespace characters.
            **extra_kwargs: Captured for provider configuration compatibility.
        """
        if isinstance(ner_provider, type):
            self.ner_provider = ner_provider()
        else:
            self.ner_provider = ner_provider

        if isinstance(splitter, type):
            self.splitter = splitter()
        else:
            self.splitter = splitter

        self.provider_kwargs = provider_kwargs or {}
        self.provider_kwargs.update(extra_kwargs)
        self.strip_markdown = strip_markdown
        self.collapse_whitespace = collapse_whitespace

    def execute(
        self, 
        data: Union[str, ChunkPayload, List[ChunkPayload]], 
        context: ExecutionContext
    ) -> List[EntityChunkPayload]:
        """
        Extracts entities from input data with strictly source-level annotations.

        The execution lifecycle ensures that the lineage established by the provider 
        (linking entities to their specific evidence chunks) is preserved and 
        not overwritten by the strategy.
        """
        if not data:
            return []

        inputs: List[ChunkPayload] = []
        if isinstance(data, list):
            inputs = data
        elif isinstance(data, ChunkPayload):
            inputs = [data]
        else:
            inputs = [ChunkPayload(content=data, content_raw=data)]

        final_results: List[EntityChunkPayload] = []

        for source in inputs:
            raw_units: List[ChunkPayload] = []
            if self.splitter:
                raw_units = self.splitter.execute(source.content, context)
            else:
                raw_units = [self._apply_annotators_to_payload(source.content, context)]

            for unit in raw_units:
                unit.content = clean_text(
                    unit.content,
                    strip_markdown=self.strip_markdown,
                    collapse_whitespace=self.collapse_whitespace
                )

            entity_payloads = self.ner_provider.extract_entities(
                raw_units, 
                **self.provider_kwargs
            )

            # Do not overwrite ep.children here. 
            # The provider is responsible for linking entities to their evidence chunks.
            final_results.extend(entity_payloads)

        return final_results


class SummaryChunking(BaseStrategy):
    """
    An advanced compound strategy that generates summaries for content units.

    This strategy orchestrates a two-stage process: first, it optionally decomposes 
    large input chunks into smaller semantic units using a splitter; then, it 
    leverages an LLM provider to summarize each resulting unit.

    It exposes configuration flags for the internal 'clean_text' utility, 
    allowing control over whitespace collapsing and markdown stripping. 
    Annotators are applied only to the final summarized output.
    """

    def __init__(
        self, 
        provider: Any, 
        splitter: Optional[BaseStrategy] = None, 
        provider_kwargs: Optional[dict] = None,
        strip_markdown: bool = False,
        collapse_whitespace: bool = True
    ):
        """
        Initializes the strategy with a provider, an optional internal splitter,
        and text cleaning configurations.

        Args:
            provider: The LLM provider instance for summarization.
            splitter: An optional strategy for internal decomposition.
            provider_kwargs: Configuration for the summarization provider.
            strip_markdown: Whether to remove markdown formatting during cleaning.
            collapse_whitespace: Whether to collapse multiple whitespaces into one.
        """
        self.provider = provider
        self.splitter = splitter
        self.provider_kwargs = provider_kwargs or {}
        self.strip_markdown = strip_markdown
        self.collapse_whitespace = collapse_whitespace

    def execute(
        self, 
        data: Union[str, ChunkPayload, List[ChunkPayload]], 
        context: ExecutionContext
    ) -> List[ChunkPayload]:
        """
        Processes the input data to generate normalized and annotated summaries.
        
        Input normalization and final results utilize 'clean_text' with the 
        configured flags for consistency.
        """
        inputs: List[ChunkPayload] = []
        if isinstance(data, list):
            inputs = data
        elif isinstance(data, ChunkPayload):
            inputs = [data]
        else:
            normalized_input = clean_text(
                data, 
                strip_markdown=self.strip_markdown, 
                collapse_whitespace=self.collapse_whitespace
            )
            inputs = [ChunkPayload(content=normalized_input, content_raw=normalized_input, annotations={})]

        results: List[ChunkPayload] = []

        for source in inputs:
            units: List[ChunkPayload] = []
            if self.splitter:
                units = self.splitter.execute(source.content, context)
            else:
                units = [source]

            for unit in units:
                unit_content = clean_text(
                    unit.content,
                    strip_markdown=self.strip_markdown,
                    collapse_whitespace=self.collapse_whitespace
                )
                
                summary_raw = self.provider.summarize(
                    unit_content, 
                    **self.provider_kwargs
                )
                
                final_summary = clean_text(
                    summary_raw,
                    strip_markdown=self.strip_markdown,
                    collapse_whitespace=self.collapse_whitespace
                )

                new_payload = self._apply_annotators_to_payload(
                    content=final_summary,
                    context=context,
                    content_raw=unit_content
                )
                
                new_payload.children = [source]
                
                results.append(new_payload)

        return results
    


class TopicBasedChunking(BaseStrategy):
    """
    An advanced compound strategy that categorizes content by identifying primary topics.

    This strategy follows a multi-stage workflow:
    1. It normalizes input data into a base ChunkPayload without early annotation.
    2. It utilizes an internal splitter to generate atomic units from the input.
    3. It cleans these atomic units and passes them as a batch to a topic provider.
    4. It establishes lineage by linking the resulting TopicChunkPayloads to the 
       atomic units that originated them.
    5. It applies late-stage annotations to the generated topic nodes.
    """

    def __init__(
        self, 
        topic_provider: Any, 
        splitter: Optional[BaseStrategy] = None, 
        provider_kwargs: Optional[dict] = None,
        strip_markdown: bool = False,
        collapse_whitespace: bool = True
    ):
        """
        Initializes the strategy with a topic provider and decomposition settings.

        Args:
            topic_provider: Object complying with TopicProviderProtocol.
            splitter: Optional strategy for internal content decomposition.
            provider_kwargs: Configuration for the topic provider's method.
            strip_markdown: Flag to remove markdown during text cleaning.
            collapse_whitespace: Flag to unify whitespace characters.
        """
        self.topic_provider = topic_provider
        self.splitter = splitter
        self.provider_kwargs = provider_kwargs or {}
        self.strip_markdown = strip_markdown
        self.collapse_whitespace = collapse_whitespace

    def execute(
        self, 
        data: Union[str, ChunkPayload, List[ChunkPayload]], 
        context: ExecutionContext
    ) -> List[TopicChunkPayload]:
        """
        Processes data into TopicChunkPayloads while preserving unit-level lineage.

        The execution process ensures that raw input is not over-cleaned before 
        splitting, allowing the splitter to detect boundaries. Once split, 
        units are cleaned and passed to the provider. The results are then 
        linked to these units and annotated.
        """
        inputs: List[ChunkPayload] = []
        if isinstance(data, list):
            inputs = data
        elif isinstance(data, ChunkPayload):
            inputs = [data]
        else:
            inputs = [ChunkPayload(content=data, content_raw=data)]

        final_results: List[TopicChunkPayload] = []

        for source in inputs:
            units: List[ChunkPayload] = []
            if self.splitter:
                units = self.splitter.execute(source.content, context)
            else:
                units = [source]

            for unit in units:
                unit.content = clean_text(
                    unit.content,
                    strip_markdown=self.strip_markdown,
                    collapse_whitespace=self.collapse_whitespace
                )

            topic_payloads = self.topic_provider.assign_topics(
                units, 
                **self.provider_kwargs
            )

            for tp in topic_payloads:
                if context and context.annotators:
                    for alias, annotator in context.annotators.items():
                        try:
                            tp.annotations[alias] = annotator.execute(tp.content, context)
                        except Exception as e:
                            tp.annotations[alias] = {"error": str(e)}
                
                final_results.append(tp)

        return final_results


class ContextualChunking(BaseStrategy):
    """
    An advanced compound strategy that enriches text fragments with situational context.

    This strategy orchestrates a workflow where input data is decomposed into 
    atomic units, which are then enriched with situational context. Following 
    strict architectural guidelines, annotations are applied exclusively to the 
    original source fragments and never to the generated context metadata.

    The strategy is fully compliant with the ContextualProviderProtocol, utilizing 
    the 'assign_context' method to interface with providers. It ensures that 
    the situational context remains a pure metadata field, while the source 
    content carries the enrichment from the pipeline's annotators.
    """

    def __init__(
        self, 
        provider: Any, 
        splitter: Optional[BaseStrategy] = None, 
        provider_kwargs: Optional[dict] = None,
        strip_markdown: bool = False,
        collapse_whitespace: bool = True,
        **extra_kwargs: Any
    ):
        """
        Initializes the strategy with a contextual provider and cleaning configurations.

        Args:
            provider: An object conforming to ContextualProviderProtocol.
            splitter: An optional internal strategy for content decomposition.
            provider_kwargs: Configuration for the contextual provider.
            strip_markdown: Flag to remove markdown during text cleaning.
            collapse_whitespace: Flag to unify whitespace characters.
            **extra_kwargs: Captured for provider configuration compatibility.
        """
        self.provider = provider
        self.splitter = splitter
        self.provider_kwargs = provider_kwargs or {}
        self.provider_kwargs.update(extra_kwargs)
        self.strip_markdown = strip_markdown
        self.collapse_whitespace = collapse_whitespace

    def execute(
        self, 
        data: Union[str, ChunkPayload, List[ChunkPayload]], 
        context: ExecutionContext
    ) -> List[ContextualChunkPayload]:
        """
        Processes data into ContextualChunkPayloads with strictly source-level annotations.

        The execution lifecycle follows these prioritized stages:
        1. Input Normalization: Standardizes input into ChunkPayloads. Raw text is 
           preserved to allow the splitter to detect structural boundaries.
        2. Content Decomposition and Annotation: The splitter generates units. 
           These units are cleaned and annotated within the splitter's own 
           execution cycle.
        3. Context Assignment: The batch of cleaned and annotated units is passed 
           to the provider's 'assign_context' method to generate descriptions.
        4. Result Finalization: The resulting ContextualChunkPayloads are returned 
           with their generated context intact and without further annotation.
        """
        inputs: List[ChunkPayload] = []
        if isinstance(data, list):
            inputs = data
        elif isinstance(data, ChunkPayload):
            inputs = [data]
        else:
            # We don't clean yet to let the splitter see original structure (\n\n, etc.)
            inputs = [ChunkPayload(content=data, content_raw=data)]

        final_results: List[ContextualChunkPayload] = []

        for source in inputs:
            units: List[ChunkPayload] = []
            if self.splitter:
                # The splitter handles annotation of the fragments internally
                units = self.splitter.execute(source.content, context)
            else:
                # If no splitter, annotate the source once here
                units = [self._apply_annotators_to_payload(source.content, context)]

            # Clean content of units using the project's utility flags
            for unit in units:
                unit.content = clean_text(
                    unit.content,
                    strip_markdown=self.strip_markdown,
                    collapse_whitespace=self.collapse_whitespace
                )

            # Provider generates the 'context' field based on the annotated units
            contextual_payloads = self.provider.assign_context(
                units, 
                **self.provider_kwargs
            )

            for cp in contextual_payloads:
                cp.children = [source]
                final_results.append(cp)

        return final_results


class ZettelkastenChunking(BaseStrategy):
    """
    An advanced compound strategy that transforms text fragments into atomic knowledge units.

    This strategy orchestrates the synthesis of raw information into atomic hypotheses. 
    Following strict architectural guidelines, annotations are applied exclusively 
    to the original source fragments (evidence) and never to the synthesized 
    hypothesis produced by the provider.

    The strategy is fully compliant with the ZettelProviderProtocol, interfacing 
    via the 'extract_zettels' method. It handles both provider instances and 
    class types for flexible instantiation.
    """

    def __init__(
        self, 
        zettel_provider: Optional[Any] = None, 
        splitter: Optional[Any] = None, 
        provider_kwargs: Optional[dict] = None,
        strip_markdown: bool = False,
        collapse_whitespace: bool = True,
        **extra_kwargs: Any
    ):
        """
        Initializes the Zettel strategy with its provider and configuration.

        Args:
            zettel_provider: The primary provider for zettel extraction.
            splitter: An optional strategy for initial content decomposition.
            provider_kwargs: Configuration for the Zettel provider.
            strip_markdown: Flag to remove markdown during text cleaning.
            collapse_whitespace: Flag to unify whitespace characters.
            **extra_kwargs: Captured for provider configuration compatibility.
        """
        self.zettel_provider = zettel_provider or extra_kwargs.get("provider")
        
        if not self.zettel_provider:
            raise TypeError("ZettelkastenChunking.__init__() missing 1 required positional argument: 'zettel_provider'")

        if isinstance(self.zettel_provider, type):
            self.zettel_provider = self.zettel_provider()
            
        if isinstance(splitter, type):
            self.splitter = splitter()
        else:
            self.splitter = splitter

        self.provider_kwargs = provider_kwargs or {}
        self.provider_kwargs.update({k: v for k, v in extra_kwargs.items() if k != "provider"})
        
        self.strip_markdown = strip_markdown
        self.collapse_whitespace = collapse_whitespace

    def execute(
        self, 
        data: Union[str, ChunkPayload, List[ChunkPayload]], 
        context: ExecutionContext
    ) -> List[ZettelChunkPayload]:
        """
        Executes the Zettel extraction logic with strictly source-level annotations.

        The execution lifecycle ensures:
        1. Normalization: Input is standardized without early annotation.
        2. Splitting and Source Annotation: The splitter generates units. Each 
           unit is cleaned and annotated based on its original content.
        3. Extraction: The provider analyzes the annotated units to extract Zettels.
        4. Hypothesis Cleaning: The resulting hypothesis is cleaned via utilities 
           but remains free of additional pipeline annotations.
        5. Lineage: Zettels link back to the source input while maintaining 
           their annotated evidence as children.
        """
        if not data:
            return []

        inputs: List[ChunkPayload] = []
        if isinstance(data, list):
            inputs = data
        elif isinstance(data, ChunkPayload):
            inputs = [data]
        else:
            inputs = [ChunkPayload(content=data, content_raw=data)]

        final_results: List[ZettelChunkPayload] = []

        for source in inputs:
            raw_units: List[Union[str, ChunkPayload]] = []
            if self.splitter:
                raw_units = self.splitter.execute(source.content, context)
            else:
                raw_units = [source]

            if not raw_units:
                continue

            processed_units: List[ChunkPayload] = []
            for unit in raw_units:
                if isinstance(unit, str):
                    clean_u = clean_text(unit, self.strip_markdown, self.collapse_whitespace)
                    processed_units.append(self._apply_annotators_to_payload(clean_u, context))
                else:
                    unit.content = clean_text(unit.content, self.strip_markdown, self.collapse_whitespace)
                    processed_units.append(unit)

            zettel_payloads = self.zettel_provider.extract_zettels(
                processed_units, 
                **self.provider_kwargs
            )

            for zp in zettel_payloads:
                zp.hypothesis = clean_text(
                    zp.hypothesis,
                    strip_markdown=self.strip_markdown,
                    collapse_whitespace=self.collapse_whitespace
                )
                
                final_results.append(zp)

        return final_results