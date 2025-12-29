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