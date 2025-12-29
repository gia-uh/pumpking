import re
from typing import List

from pumpking.models import ChunkPayload
from pumpking.protocols import ExecutionContext
from pumpking.strategies.base import BaseStrategy


class RegexChunking(BaseStrategy):
    SUPPORTED_INPUTS = [str]
    PRODUCED_OUTPUT = List[ChunkPayload]

    def __init__(self, pattern: str) -> None:
        self.pattern = pattern

    def execute(self, data: str, context: ExecutionContext) -> List[ChunkPayload]:
        if not data:
            return []

        # Split content
        chunks = re.split(self.pattern, data)
        # Filter empty strings resulting from split artifacts
        valid_chunks = [chunk for chunk in chunks if chunk]

        # Use the helper to wrap strings into ChunkPayloads and apply annotations
        return self._apply_annotators_to_list(valid_chunks, context)
    
class FixedSizeChunking(BaseStrategy):
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
            chunk = data[start:end]
            chunks.append(chunk)

            start += self.chunk_size - self.overlap
        
        return self._apply_annotators_to_list(chunks, context)