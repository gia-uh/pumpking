import re
from typing import List

from pumpking.models import ChunkPayload
from pumpking.utils import clean_text
from pumpking.protocols import ExecutionContext
from pumpking.strategies.base import BaseStrategy


class RegexChunking(BaseStrategy):
    """
    Splits text based on a provided regular expression pattern.
    
    This strategy splits the raw text first, then cleans the resulting chunks.
    This preserves the split pattern's structural integrity.
    """
    SUPPORTED_INPUTS = [str]
    PRODUCED_OUTPUT = List[ChunkPayload]

    def __init__(self, pattern: str) -> None:
        self.pattern = pattern

    def execute(self, data: str, context: ExecutionContext) -> List[ChunkPayload]:
        if not data:
            return []

        raw_chunks = re.split(self.pattern, data)
        
        payloads = []
        for raw_chunk in raw_chunks:
            if not raw_chunk:
                continue
                
            cleaned_content = clean_text(raw_chunk, collapse_whitespace=False).strip()
            
            if cleaned_content:
                payload = self._apply_annotators_to_payload(
                    content=cleaned_content,
                    context=context,
                    content_raw=raw_chunk
                )
                payloads.append(payload)

        return payloads


class FixedSizeChunking(BaseStrategy):
    """
    Splits text into fixed-size segments with optional overlap.
    
    This strategy operates on the raw text to ensure exact character counts,
    cleaning each chunk individually afterwards.
    """
    SUPPORTED_INPUTS = [str]
    PRODUCED_OUTPUT = List[ChunkPayload]

    def __init__(self, chunk_size: int, overlap: int = 0) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be strictly less than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def execute(self, data: str, context: ExecutionContext) -> List[ChunkPayload]:
        if not data:
            return []
            
        chunks = []
        start = 0
        text_len = len(data)

        while start < text_len:
            end = start + self.chunk_size
            
            raw_chunk = data[start:end]
            cleaned_chunk = clean_text(raw_chunk)
            
            if cleaned_chunk:
                payload = self._apply_annotators_to_payload(
                    content=cleaned_chunk,
                    context=context,
                    content_raw=raw_chunk
                )
                chunks.append(payload)

            start += self.chunk_size - self.overlap
        
        return chunks

class ParagraphChunking(RegexChunking):
    """
    Specialized RegexChunking that splits text by double newlines.
    Useful for separating paragraphs, headers from body text, etc.
    """
    def __init__(self) -> None:
        super().__init__(pattern=r"\n\n+")
        
class SentenceChunking(RegexChunking):
    """
    Splits text into sentences based on punctuation boundaries using a lookbehind pattern.
    """
    def __init__(self) -> None:
        super().__init__(pattern=r"(?<=[.!?])\s+")
        
class SlidingWindowChunking(BaseStrategy):
    """
    Splits text into fixed-size word windows with overlap.
    """
    def __init__(self, window_size: int, overlap: int) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= window_size:
            raise ValueError("overlap must be strictly less than window_size")

        self.window_size = window_size
        self.overlap = overlap

    def execute(self, data: str, context: ExecutionContext) -> List[ChunkPayload]:
        if not data:
            return []

        words = data.split()
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
                    content=cleaned_content,
                    context=context,
                    content_raw=raw_content
                )
                chunks.append(payload)
            
            if i + self.window_size >= len(words):
                break

        return chunks
    
class AdaptiveChunking(RegexChunking):
    """
    Groups sentences into chunks within a character limit range.

    This strategy leverages regex splitting to identify sentence boundaries
    and then aggregates them into larger chunks. It prevents running annotators
    on individual sentences by using an empty execution context for the
    initial split, applying them only to the final aggregated chunks.
    """
    def __init__(self, min_chunk_size: int, max_chunk_size: int) -> None:
        super().__init__(pattern=r"(?<=[.!?])\s+")
        if min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be positive")
        if max_chunk_size < min_chunk_size:
            raise ValueError("max_chunk_size must be greater than or equal to min_chunk_size")

        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def execute(self, data: str, context: ExecutionContext) -> List[ChunkPayload]:
        empty_context = ExecutionContext()
        sentences = super().execute(data, empty_context)
        
        chunks = []
        current_buffer = []
        current_len = 0

        for sentence_payload in sentences:
            content = sentence_payload.content
            add_len = len(content)
            sep_len = 1 if current_buffer else 0
            
            projected_len = current_len + sep_len + add_len

            if projected_len > self.max_chunk_size and current_buffer:
                self._emit_buffer(chunks, current_buffer, context)
                current_buffer = [content]
                current_len = add_len
            else:
                current_buffer.append(content)
                current_len = projected_len

        if current_buffer:
            self._emit_buffer(chunks, current_buffer, context)

        return chunks

    def _emit_buffer(self, chunks: List[ChunkPayload], buffer: List[str], context: ExecutionContext) -> None:
        raw_text = " ".join(buffer)
        cleaned_content = clean_text(raw_text)
        
        if cleaned_content:
            payload = self._apply_annotators_to_payload(
                content=cleaned_content,
                context=context,
                content_raw=raw_text
            )
            chunks.append(payload)