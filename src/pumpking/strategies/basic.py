import re
from typing import List, Any, Union

from pumpking.models import ChunkPayload
from pumpking.utils import clean_text
from pumpking.protocols import ExecutionContext
from pumpking.strategies.base import BaseStrategy


class RegexChunking(BaseStrategy):
    """
    A foundational strategy that divides text input based on a configurable regular
    expression pattern.

    This class serves as a versatile splitter capable of handling various structured
    text formats defined by specific delimiters. It extends the base strategy to
    include options for whitespace normalization, ensuring that the resulting segments
    are clean and ready for downstream processing.
    """

    def __init__(self, pattern: str, collapse_whitespace: bool = True) -> None:
        """
        Initializes the regex-based splitter with a specific pattern and cleaning configuration.

        Args:
            pattern: The regular expression string used to identify split boundaries.
            collapse_whitespace: A flag indicating whether to reduce multiple whitespace
                                 characters into a single space in the resulting chunks.
        """
        self.pattern = pattern
        self.collapse_whitespace = collapse_whitespace

    def execute(
        self, data: Union[str, ChunkPayload], context: ExecutionContext
    ) -> List[ChunkPayload]:
        """
        Executes the splitting logic using the compiled regex pattern.

        The method handles input normalization, regex splitting, and post-processing
        cleaning. It ensures that empty results are filtered out and that valid
        segments are wrapped in ChunkPayload objects with annotations applied.

        Args:
            data: The input text string or ChunkPayload to be processed.
            context: The shared execution context containing global state and annotators.

        Returns:
            A list of ChunkPayload objects representing the split segments.
        """
        if not data:
            return []

        text_content = data.content if isinstance(data, ChunkPayload) else str(data)

        if not text_content:
            return []

        raw_chunks = re.split(self.pattern, text_content)
        payloads = []

        for raw_chunk in raw_chunks:
            if not raw_chunk:
                continue

            cleaned_content = clean_text(
                raw_chunk, collapse_whitespace=self.collapse_whitespace
            ).strip()

            if cleaned_content:
                payload = self._apply_annotators_to_payload(
                    content=cleaned_content, context=context, content_raw=raw_chunk
                )
                payloads.append(payload)

        return payloads


class FixedSizeChunking(BaseStrategy):
    """
    A rigid strategy that segments text into windows of a fixed character length.

    This approach is useful for processing text where semantic boundaries are less
    critical than strict size constraints. It supports an overlap parameter to ensure
    continuity between adjacent chunks, mitigating the loss of context at the edges.
    """

    def __init__(self, chunk_size: int, overlap: int = 0) -> None:
        """
        Initializes the fixed-size splitter with validation for size and overlap parameters.

        Args:
            chunk_size: The exact number of characters for each text segment.
            overlap: The number of characters to repeat at the beginning of the next chunk.

        Raises:
            ValueError: If chunk_size is not positive, if overlap is negative, or if
                        overlap is greater than or equal to chunk_size.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be strictly less than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def execute(
        self, data: Union[str, ChunkPayload], context: ExecutionContext
    ) -> List[ChunkPayload]:
        """
        Slices the input text into fixed-size segments.

        The method iterates through the text using a sliding window approach derived
        from the chunk size and overlap. Each extracted segment is cleaned and
        converted into a ChunkPayload.

        Args:
            data: The input text or payload.
            context: The execution context.

        Returns:
            A list of ChunkPayloads of uniform length (except potentially the last one).
        """
        if not data:
            return []

        text_content = data.content if isinstance(data, ChunkPayload) else str(data)

        if not text_content:
            return []

        chunks = []
        start = 0
        text_len = len(text_content)

        while start < text_len:
            end = start + self.chunk_size

            raw_chunk = text_content[start:end]
            cleaned_chunk = clean_text(raw_chunk)

            if cleaned_chunk:
                payload = self._apply_annotators_to_payload(
                    content=cleaned_chunk, context=context, content_raw=raw_chunk
                )
                chunks.append(payload)

            start += self.chunk_size - self.overlap

        return chunks


class ParagraphChunking(RegexChunking):
    """
    A specialized implementation of RegexChunking designed to split text by paragraphs.

    This class pre-configures the regex pattern to identify double newline sequences,
    which are the standard delimiters for paragraphs in most plain text formats.
    It intentionally disables aggressive whitespace collapsing to preserve the
    internal formatting of the paragraph, such as line breaks in lists or poetry.
    """

    def __init__(self) -> None:
        """
        Initializes the paragraph splitter using a double-newline regex pattern.
        """
        super().__init__(pattern=r"\n\n+", collapse_whitespace=False)


class SentenceChunking(RegexChunking):
    """
    A specialized implementation of RegexChunking designed to split text into sentences.

    This class utilizes a regular expression with lookbehind assertions to identify
    sentence boundaries marked by punctuation (periods, exclamation marks, question marks)
    followed by whitespace. This ensures that the punctuation remains attached to the
    preceding sentence. Whitespace collapsing is enabled to produce normalized output.
    """

    def __init__(self) -> None:
        """
        Initializes the sentence splitter using a punctuation-based regex pattern.
        """
        super().__init__(pattern=r"(?<=[.!?])\s+", collapse_whitespace=True)


class SlidingWindowChunking(BaseStrategy):
    """
    A strategy that splits text into overlapping windows based on word count.

    Unlike fixed-size chunking which operates on characters, this strategy respects
    word boundaries, creating chunks that contain a specific number of words.
    It is particularly effective for generating context-rich segments for embedding
    models where maintaining complete words is essential.
    """

    def __init__(self, window_size: int, overlap: int) -> None:
        """
        Initializes the sliding window splitter with validation.

        Args:
            window_size: The number of words to include in each chunk.
            overlap: The number of words to overlap between consecutive chunks.

        Raises:
            ValueError: If parameters are invalid (e.g., negative overlap, or
                        overlap exceeding window size).
        """
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= window_size:
            raise ValueError("overlap must be strictly less than window_size")

        self.window_size = window_size
        self.overlap = overlap

    def execute(
        self, data: Union[str, ChunkPayload], context: ExecutionContext
    ) -> List[ChunkPayload]:
        """
        Splits the input text into word-based windows.

        The method tokenizes the text into words and then iterates over them using
        the calculated stride. Each window of words is joined back into a string,
        cleaned, and wrapped in a payload.

        Args:
            data: The input text or payload.
            context: The execution context.

        Returns:
            A list of ChunkPayloads representing the word windows.
        """
        if not data:
            return []

        text_content = data.content if isinstance(data, ChunkPayload) else str(data)

        if not text_content:
            return []

        words = text_content.split()
        if not words:
            return []

        step = self.window_size - self.overlap
        chunks = []

        for i in range(0, len(words), step):
            window = words[i : i + self.window_size]
            raw_content = " ".join(window)

            cleaned_content = clean_text(raw_content)

            if cleaned_content:
                payload = self._apply_annotators_to_payload(
                    content=cleaned_content, context=context, content_raw=raw_content
                )
                chunks.append(payload)

            if i + self.window_size >= len(words):
                break

        return chunks


class AdaptiveChunking(RegexChunking):
    """
    An advanced strategy that dynamically groups sentences into chunks to fit within
    size constraints.

    This class leverages the inheritance from RegexChunking to first split the text
    into sentences. It then intelligently accumulates these sentences into buffers,
    emitting a chunk only when the buffer reaches a specified maximum character limit.
    It ensures that chunks are semantically coherent (sentences are not split) and
    preserves lineage by linking the original sentence payloads as children.
    """

    def __init__(self, min_chunk_size: int, max_chunk_size: int) -> None:
        """
        Initializes the adaptive chunker with size constraints.

        Args:
            min_chunk_size: The soft lower limit for chunk size (not strictly enforced
                            if a single sentence is smaller, but used for logic guidance).
            max_chunk_size: The hard upper limit for chunk size.

        Raises:
            ValueError: If size parameters are invalid.
        """
        super().__init__(pattern=r"(?<=[.!?])\s+", collapse_whitespace=True)
        if min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be positive")
        if max_chunk_size < min_chunk_size:
            raise ValueError(
                "max_chunk_size must be greater than or equal to min_chunk_size"
            )

        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def execute(
        self, data: Union[str, ChunkPayload], context: ExecutionContext
    ) -> List[ChunkPayload]:
        """
        Executes the adaptive grouping logic.

        It first decomposes the input into sentences using the parent class's logic.
        Then, it iterates through the sentences, accumulating them into a buffer until
        adding another sentence would exceed 'max_chunk_size'. At that point, the
        buffer is emitted as a new chunk.

        Args:
            data: The input text or payload.
            context: The execution context.

        Returns:
            A list of ChunkPayloads, where each payload is an aggregate of sentences.
        """
        empty_context = ExecutionContext()
        sentences = super().execute(data, empty_context)

        chunks = []
        current_buffer: List[ChunkPayload] = []
        current_len = 0

        for sentence_payload in sentences:
            content = sentence_payload.content
            if not content:
                continue

            add_len = len(content)
            sep_len = 1 if current_buffer else 0

            projected_len = current_len + sep_len + add_len

            if projected_len > self.max_chunk_size and current_buffer:
                self._emit_buffer(chunks, current_buffer, context)
                current_buffer = [sentence_payload]
                current_len = add_len
            else:
                current_buffer.append(sentence_payload)
                current_len = projected_len

        if current_buffer:
            self._emit_buffer(chunks, current_buffer, context)

        return chunks

    def _emit_buffer(
        self,
        chunks: List[ChunkPayload],
        buffer: List[ChunkPayload],
        context: ExecutionContext,
    ) -> None:
        """
        Internal helper to construct a combined ChunkPayload from a buffer of sentences.

        It joins the content of the buffered payloads, cleans the result, applies
        annotators, and critically, assigns the original sentence payloads to the
        'children' field of the new payload to maintain the graph lineage.

        Args:
            chunks: The list to append the new payload to.
            buffer: The list of source sentence payloads.
            context: The execution context.
        """
        raw_text = " ".join([p.content for p in buffer])
        cleaned_content = clean_text(raw_text)

        if cleaned_content:
            payload = self._apply_annotators_to_payload(
                content=cleaned_content, context=context, content_raw=raw_text
            )
            payload.children = buffer
            chunks.append(payload)
