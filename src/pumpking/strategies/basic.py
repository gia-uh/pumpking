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