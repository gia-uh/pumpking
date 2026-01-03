import markdown
from html.parser import HTMLParser
from typing import List, Optional, Any, Union, Type, Dict
from pumpking.models import (
    ChunkPayload,
    EntityChunkPayload,
    TopicChunkPayload,
    ContextualChunkPayload,
    ZettelChunkPayload,
)
from pumpking.protocols import (
    ExecutionContext,
    NERProviderProtocol,
    SummaryProviderProtocol,
    TopicProviderProtocol,
    ContextualProviderProtocol,
    ZettelProviderProtocol,
)
from pumpking.strategies.base import BaseStrategy
from pumpking.strategies.basic import (
    SentenceChunking,
    AdaptiveChunking,
    ParagraphChunking,
)
from pumpking.utils import clean_text


class _SectionNode:
    """
    Internal data structure used to represent a logical section within a document's hierarchy.

    This node acts as a container for a specific document section, holding metadata
    such as the header level (depth) and title, as well as accumulating the raw text
    content associated with that section. It supports a recursive structure where
    each node can maintain a list of child nodes (subsections), enabling the
    construction of a complete document tree.
    """

    def __init__(self, level: int, title: str = "") -> None:
        """
        Initializes a new section node.

        Args:
            level: The hierarchical depth of the section (e.g., 1 for H1, 2 for H2).
                   Level 0 is typically reserved for the document root.
            title: The extracted text of the header defining this section.
        """
        self.level = level
        self.title = title
        self.content_buffer: List[str] = []
        self.children: List["_SectionNode"] = []

    def get_text_content(self) -> str:
        """
        Retrieves the aggregated text content of this section.

        Returns:
            The combined string of all text data accumulated in the content buffer.
        """
        return "".join(self.content_buffer)


class _MarkdownStructureParser(HTMLParser):
    """
    A specialized HTML parser designed to reconstruct a document's hierarchy from
    rendered Markdown.

    This class parses the HTML output of a Markdown converter to build a tree of
    _SectionNodes. It tracks the depth of headers (H1-H6) to manage the nesting
    of sections on a stack. Content found between headers is attributed to the
    currently active section on the top of the stack.
    """
    
    def __init__(self) -> None:
        """
        Initializes the parser state.

        Sets up the root node (level 0) and initializes the stack with the root
        as the active context. The 'in_header' flag tracks whether the parser
        is currently processing a header tag to capture titles correctly.
        """
        super().__init__()
        self.root = _SectionNode(level=0)
        self.stack: List[_SectionNode] = [self.root]
        self.in_header = False

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> None:
        """
        Handles the opening of HTML tags to detect structure.

        When a header tag (h1-h6) is encountered, the stack is unwound until the
        current level is less than the new header's level, ensuring correct nesting.
        A new _SectionNode is then created and pushed onto the stack.

        Args:
            tag: The name of the tag (e.g., 'h1', 'p', 'div').
            attrs: A list of (name, value) pairs containing the attributes found
                   inside the tag's <> brackets.
        """
        if tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            level = int(tag[1])
            self.in_header = True
            while self.stack[-1].level >= level:
                self.stack.pop()
            new_node = _SectionNode(level=level)
            self.stack[-1].children.append(new_node)
            self.stack.append(new_node)

    def handle_endtag(self, tag: str) -> None:
        """
        Handles the closing of HTML tags.

        If a header tag closes, the 'in_header' flag is reset, and the title
        of the current section is cleaned to remove any internal markup artifacts.

        Args:
            tag: The name of the tag being closed.
        """
        if tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            self.in_header = False
            self.stack[-1].title = clean_text(self.stack[-1].title)

    def handle_data(self, data: str) -> None:
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
from pumpking.utils import clean_text


class _SectionNode:
    """
    Internal data structure used to represent a logical section within a document's hierarchy.

    This node acts as a container for a specific document section, holding metadata
    such as the header level (depth) and title, as well as accumulating the raw text
    content associated with that section. It supports a recursive structure where
    each node can maintain a list of child nodes (subsections), enabling the
    construction of a complete document tree.
    """

    def __init__(self, level: int, title: str = "") -> None:
        """
        Initializes a new section node.

        Args:
            level: The hierarchical depth of the section (e.g., 1 for H1, 2 for H2).
                   Level 0 is typically reserved for the document root.
            title: The extracted text of the header defining this section.
        """
        self.level = level
        self.title = title
        self.content_buffer: List[str] = []
        self.children: List["_SectionNode"] = []

    def get_text_content(self) -> str:
        """
        Retrieves the aggregated text content of this section.

        Returns:
            The combined string of all text data accumulated in the content buffer.
        """
        return "".join(self.content_buffer)

class _MarkdownStructureParser(HTMLParser):
    """
    A specialized HTML parser designed to reconstruct a document's hierarchy from
    rendered Markdown.

    This class parses the HTML output of a Markdown converter to build a tree of
    _SectionNodes. It tracks the depth of headers (H1-H6) to manage the nesting
    of sections on a stack. Content found between headers is attributed to the
    currently active section on the top of the stack.
    """

    def __init__(self) -> None:
        """
        Initializes the parser state.

        Sets up the root node (level 0) and initializes the stack with the root
        as the active context. The 'in_header' flag tracks whether the parser
        is currently processing a header tag to capture titles correctly.
        """
        super().__init__()
        self.root = _SectionNode(level=0)
        self.stack: List[_SectionNode] = [self.root]
        self.in_header = False

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> None:
        """
        Handles the opening of HTML tags to detect structure.

        When a header tag (h1-h6) is encountered, the stack is unwound until the
        current level is less than the new header's level, ensuring correct nesting.
        A new _SectionNode is then created and pushed onto the stack.

        Args:
            tag: The name of the tag (e.g., 'h1', 'p', 'div').
            attrs: A list of (name, value) pairs containing the attributes found
                   inside the tag's <> brackets.
        """
        if tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            level = int(tag[1])
            self.in_header = True
            while self.stack[-1].level >= level:
                self.stack.pop()
            new_node = _SectionNode(level=level)
            self.stack[-1].children.append(new_node)
            self.stack.append(new_node)

    def handle_endtag(self, tag: str) -> None:
        """
        Handles the closing of HTML tags.

        If a header tag closes, the 'in_header' flag is reset, and the title
        of the current section is cleaned to remove any internal markup artifacts.

        Args:
            tag: The name of the tag being closed.
        """
        if tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            self.in_header = False
            self.stack[-1].title = clean_text(self.stack[-1].title)

    def handle_data(self, data: str) -> None:
        """
        Processes text data found between tags.

        If the parser is inside a header tag, the data is appended to the current
        section's title. Otherwise, it is treated as body content and appended
        to the content buffer of the active section.

        Args:
            data: The raw text string.
        """
        if self.in_header:
            self.stack[-1].title += data
        else:
            self.stack[-1].content_buffer.append(data)


class HierarchicalChunking(BaseStrategy):
    """
    A structural strategy that parses Markdown to produce a nested tree of chunks.

    This strategy respects the semantic structure of a document defined by its headers.
    It converts the input Markdown into a tree where each node represents a section
    (defined by a header) and contains its subsections as children.

    Unlike flat chunking strategies, this approach preserves the parent-child
    relationships inherent in the document structure. Additionally, it allows for
    a chain of sub-strategies to be applied to the text content of each section,
    enabling granular processing (e.g., splitting a section's body into sentences)
    while maintaining the macro-level hierarchy.
    """

    def __init__(self, strategies: List[BaseStrategy]) -> None:
        """
        Initializes the hierarchical strategy.

        Args:
            strategies: A list of BaseStrategy instances to be applied to the
                        body text of every identified section. These sub-strategies
                        process the leaf content within the structural tree.
        """
        self.strategies = strategies

    def execute(
        self, data: Union[str, ChunkPayload], context: ExecutionContext
    ) -> List[ChunkPayload]:
        """
        Executes the hierarchical parsing and chunking process.

        The workflow involves:
        1. Converting the input text (Markdown) to HTML.
        2. Parsing the HTML to build a structural tree of _SectionNodes.
        3. Traversing the tree to convert nodes into ChunkPayloads.
        4. Applying configured sub-strategies to the body text of each section.

        Args:
            data: The input markdown string or a ChunkPayload containing it.
            context: The execution context.

        Returns:
            A list of top-level ChunkPayloads representing the root sections of
            the document, each containing their nested children.
        """
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
        """
        Recursively transforms a _SectionNode into a ChunkPayload.

        This method:
        1. Processes the node's text content using the configured sub-strategies.
        2. Recursively calls itself for all children of the node.
        3. Constructs a new ChunkPayload encapsulating the section, linking
           both the processed body chunks and the subsections as children.
        4. Applies annotators to the newly created payload.

        Args:
            node: The _SectionNode to convert.
            context: The execution context.

        Returns:
            A ChunkPayload representing the section and its subtree.
        """
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
        """
        Applies the configured chain of sub-strategies to a segment of text.

        Currently applies the first strategy in the list.

        Args:
            content: The text content to process.
            context: The execution context.

        Returns:
            A list of ChunkPayloads generated by the sub-strategy.
        """
        if not self.strategies:
            return []

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

    Default Behavior:
        If no splitter is provided, this strategy defaults to using 'SentenceChunking'.
        This ensures that entities are detected within the context of individual sentences,
        which is the standard granularity for most NER tasks.
    """

    def __init__(
        self,
        ner_provider: NERProviderProtocol,
        splitter: Optional[BaseStrategy] = None,
        provider_kwargs: Optional[dict] = None,
        strip_markdown: bool = False,
        collapse_whitespace: bool = True,
        **extra_kwargs: Any,
    ):
        """
        Initializes the entity strategy with a specific provider and optional splitter.

        Args:
            ner_provider: An instance conforming to NERProviderProtocol.
            splitter: An optional strategy for content decomposition. If None,
                'SentenceChunking' is instantiated and used by default.
            provider_kwargs: Configuration for the NER provider.
            strip_markdown: Flag to remove markdown during text cleaning.
            collapse_whitespace: Flag to unify whitespace characters.
            **extra_kwargs: Captured for provider configuration compatibility.
        """
        self.ner_provider = ner_provider

        if splitter:
            self.splitter = splitter
        else:
            self.splitter = SentenceChunking()

        self.provider_kwargs = provider_kwargs or {}
        self.provider_kwargs.update(extra_kwargs)
        self.strip_markdown = strip_markdown
        self.collapse_whitespace = collapse_whitespace

    def execute(
        self,
        data: Union[str, ChunkPayload, List[ChunkPayload]],
        context: ExecutionContext,
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
                    collapse_whitespace=self.collapse_whitespace,
                )

            entity_payloads = self.ner_provider.extract_entities(
                raw_units, **self.provider_kwargs
            )

            final_results.extend(entity_payloads)

        return final_results


class SummaryChunking(BaseStrategy):
    """
    An advanced compound strategy that generates summaries for content units.

    This strategy orchestrates a two-stage process: first, it decomposes large input
    chunks into smaller semantic units using a splitter; then, it leverages an LLM
    provider to summarize each resulting unit.

    Default Behavior:
        If no splitter is provided, this strategy defaults to using 'AdaptiveChunking'
        configured with common values (min_chunk_size=1000, max_chunk_size=3000).
        This default ensures that the text is grouped into semantically coherent blocks
        of sufficient size for meaningful summarization, rather than summarizing individual
        sentences or arbitrary fragments.

        If a user requires different sizing parameters, they should explicitly instantiate
        and pass their own 'AdaptiveChunking' (or other strategy) instance to the constructor.
    """

    def __init__(
        self,
        provider: SummaryProviderProtocol,
        splitter: Optional[BaseStrategy] = None,
        provider_kwargs: Optional[dict] = None,
        strip_markdown: bool = False,
        collapse_whitespace: bool = True,
    ):
        """
        Initializes the strategy with a provider, an optional internal splitter,
        and text cleaning configurations.

        Args:
            provider: The LLM provider instance for summarization.
            splitter: An optional strategy for internal decomposition. If None,
                defaults to 'AdaptiveChunking(1000, 3000)'. To use custom chunk sizes,
                pass a pre-configured AdaptiveChunking instance here.
            provider_kwargs: Configuration for the summarization provider.
            strip_markdown: Whether to remove markdown formatting during cleaning.
            collapse_whitespace: Whether to collapse multiple whitespaces into one.
        """
        self.provider = provider

        if splitter:
            self.splitter = splitter
        else:
            self.splitter = AdaptiveChunking(min_chunk_size=1000, max_chunk_size=3000)

        self.provider_kwargs = provider_kwargs or {}
        self.strip_markdown = strip_markdown
        self.collapse_whitespace = collapse_whitespace

    def execute(
        self,
        data: Union[str, ChunkPayload, List[ChunkPayload]],
        context: ExecutionContext,
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
                collapse_whitespace=self.collapse_whitespace,
            )
            inputs = [
                ChunkPayload(
                    content=normalized_input,
                    content_raw=normalized_input,
                    annotations={},
                )
            ]

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
                    collapse_whitespace=self.collapse_whitespace,
                )

                summary_raw = self.provider.summarize(
                    unit_content, **self.provider_kwargs
                )

                final_summary = clean_text(
                    summary_raw,
                    strip_markdown=self.strip_markdown,
                    collapse_whitespace=self.collapse_whitespace,
                )

                new_payload = self._apply_annotators_to_payload(
                    content=final_summary, context=context, content_raw=unit_content
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

    Default Behavior:
        If no splitter is provided, this strategy defaults to using 'ParagraphChunking'.
        This assumes that paragraphs are the most natural unit of thought for detecting
        distinct topics in a document.
    """

    def __init__(
        self,
        topic_provider: TopicProviderProtocol,
        splitter: Optional[BaseStrategy] = None,
        provider_kwargs: Optional[dict] = None,
        strip_markdown: bool = False,
        collapse_whitespace: bool = True,
    ):
        """
        Initializes the strategy with a topic provider and decomposition settings.

        Args:
            topic_provider: Object complying with TopicProviderProtocol.
            splitter: Optional strategy for internal content decomposition. If None,
                defaults to 'ParagraphChunking'.
            provider_kwargs: Configuration for the topic provider's method.
            strip_markdown: Flag to remove markdown during text cleaning.
            collapse_whitespace: Flag to unify whitespace characters.
        """
        self.topic_provider = topic_provider

        if splitter:
            self.splitter = splitter
        else:
            self.splitter = ParagraphChunking()

        self.provider_kwargs = provider_kwargs or {}
        self.strip_markdown = strip_markdown
        self.collapse_whitespace = collapse_whitespace

    def execute(
        self,
        data: Union[str, ChunkPayload, List[ChunkPayload]],
        context: ExecutionContext,
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
                    collapse_whitespace=self.collapse_whitespace,
                )

            topic_payloads = self.topic_provider.assign_topics(
                units, **self.provider_kwargs
            )

            for tp in topic_payloads:
                if context and context.annotators:
                    for alias, annotator in context.annotators.items():
                        try:
                            tp.annotations[alias] = annotator.execute(
                                tp.content, context
                            )
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

    Default Behavior:
        If no splitter is provided, this strategy defaults to using 'AdaptiveChunking'
        configured with common values (min_chunk_size=1000, max_chunk_size=3000).
        This default assumes that contextual enrichment is most effective when applied
        to substantial, coherent blocks of text rather than tiny fragments.

        Users desiring different chunk sizes must provide their own pre-configured
        'AdaptiveChunking' instance.
    """

    def __init__(
        self,
        provider: ContextualProviderProtocol,
        splitter: Optional[BaseStrategy] = None,
        provider_kwargs: Optional[dict] = None,
        strip_markdown: bool = False,
        collapse_whitespace: bool = True,
        **extra_kwargs: Any,
    ):
        """
        Initializes the strategy with a contextual provider and cleaning configurations.

        Args:
            provider: An object conforming to ContextualProviderProtocol.
            splitter: An optional internal strategy for content decomposition. If None,
                defaults to 'AdaptiveChunking(1000, 3000)'.
            provider_kwargs: Configuration for the contextual provider.
            strip_markdown: Flag to remove markdown during text cleaning.
            collapse_whitespace: Flag to unify whitespace characters.
            **extra_kwargs: Captured for provider configuration compatibility.
        """
        self.provider = provider

        if splitter:
            self.splitter = splitter
        else:
            self.splitter = AdaptiveChunking(min_chunk_size=1000, max_chunk_size=3000)

        self.provider_kwargs = provider_kwargs or {}
        self.provider_kwargs.update(extra_kwargs)
        self.strip_markdown = strip_markdown
        self.collapse_whitespace = collapse_whitespace

    def execute(
        self,
        data: Union[str, ChunkPayload, List[ChunkPayload]],
        context: ExecutionContext,
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
            inputs = [ChunkPayload(content=data, content_raw=data)]

        final_results: List[ContextualChunkPayload] = []

        for source in inputs:
            units: List[ChunkPayload] = []
            if self.splitter:
                units = self.splitter.execute(source.content, context)
            else:
                units = [self._apply_annotators_to_payload(source.content, context)]

            for unit in units:
                unit.content = clean_text(
                    unit.content,
                    strip_markdown=self.strip_markdown,
                    collapse_whitespace=self.collapse_whitespace,
                )

            contextual_payloads = self.provider.assign_context(
                units, **self.provider_kwargs
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

    Default Behavior:
        If no splitter is provided, this strategy defaults to using 'ParagraphChunking'.
        The Zettelkasten method traditionally operates on discrete ideas which often
        map well to paragraph boundaries.
    """

    def __init__(
        self,
        zettel_provider: ZettelProviderProtocol,
        splitter: Optional[BaseStrategy] = None,
        provider_kwargs: Optional[dict] = None,
        strip_markdown: bool = False,
        collapse_whitespace: bool = True,
        **extra_kwargs: Any,
    ):
        """
        Initializes the Zettel strategy with its provider and configuration.

        Args:
            zettel_provider: The primary provider for zettel extraction.
            splitter: An optional strategy for initial content decomposition. If None,
                defaults to 'ParagraphChunking'.
            provider_kwargs: Configuration for the Zettel provider.
            strip_markdown: Flag to remove markdown during text cleaning.
            collapse_whitespace: Flag to unify whitespace characters.
            **extra_kwargs: Captured for provider configuration compatibility.
        """
        self.zettel_provider = zettel_provider

        if splitter:
            self.splitter = splitter
        else:
            self.splitter = ParagraphChunking()

        self.provider_kwargs = provider_kwargs or {}
        self.provider_kwargs.update(
            {k: v for k, v in extra_kwargs.items() if k != "provider"}
        )

        self.strip_markdown = strip_markdown
        self.collapse_whitespace = collapse_whitespace

    def execute(
        self,
        data: Union[str, ChunkPayload, List[ChunkPayload]],
        context: ExecutionContext,
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
                    clean_u = clean_text(
                        unit, self.strip_markdown, self.collapse_whitespace
                    )
                    processed_units.append(
                        self._apply_annotators_to_payload(clean_u, context)
                    )
                else:
                    unit.content = clean_text(
                        unit.content, self.strip_markdown, self.collapse_whitespace
                    )
                    processed_units.append(unit)

            zettel_payloads = self.zettel_provider.extract_zettels(
                processed_units, **self.provider_kwargs
            )

            for zp in zettel_payloads:
                zp.hypothesis = clean_text(
                    zp.hypothesis,
                    strip_markdown=self.strip_markdown,
                    collapse_whitespace=self.collapse_whitespace,
                )

                final_results.append(zp)

        return final_results
