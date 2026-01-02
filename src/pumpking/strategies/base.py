from typing import Any, List, Optional, Union
from pumpking.models import ChunkPayload
from pumpking.protocols import StrategyProtocol, ExecutionContext


class BaseStrategy(StrategyProtocol):
    """
    Base class for all strategies, providing core infrastructure for payload creation
    and annotation injection.

    This class serves as the foundational layer for the Strategy pattern in the
    Pumpking architecture. It enforces the separation of concerns by strictly
    producing ChunkPayload objects (intermediate data representations) and avoiding
    any coupling with the Graph storage layer (ChunkNodes).

    Strategies inheriting from this class are responsible for transforming raw input
    data into structured payloads. This base class provides the shared logic to ensure
    that all payloads are consistently instantiated and automatically processed by
    any configured annotators in the ExecutionContext.
    """

    def execute(self, data: Union[str, ChunkPayload, List[ChunkPayload]], context: ExecutionContext) -> List[ChunkPayload]:
        """
        Abstract method that must be implemented by all concrete strategies.

        This method defines the specific processing logic of the strategy.
        It receives input data which can vary in type to support different
        processing paradigms:
        - str: Raw text input for initial processing.
        - ChunkPayload: A single processed object for sequential refinement (Chain).
        - List[ChunkPayload]: A batch of objects for optimized group processing (Batching).

        Concrete implementations should strictly define their supported input types
        via type hints to allow the Pipeline orchestrator to correctly dispatch
        data (either by iterating or passing the full batch).

        Args:
            data: The input data to be processed.
            context: The shared execution context containing configuration and annotators.

        Returns:
            The processed result, consistently returning a list of ChunkPayloads.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Strategies must implement execute method.")

    def _apply_annotators_to_payload(
        self, 
        content: str, 
        context: ExecutionContext, 
        content_raw: Optional[str] = None
    ) -> ChunkPayload:
        """
        Factory method that constructs a ChunkPayload and applies all active annotators.

        This method centralizes the object creation lifecycle to guarantee consistency.
        It first instantiates a ChunkPayload with the provided content (which may be
        processed text like a summary) and the raw content (the original source).
        
        It implements a storage optimization logic: if the provided 'content_raw' is
        identical to the 'content' (or if it is not provided), the 'content_raw' field
        in the payload is set to None. This prevents data duplication when the source
        and the result are the same string.

        After instantiation, it iterates through the annotators defined in the
        ExecutionContext. Each annotator is executed against the 'content' field
        of the payload. This ensures that metadata (such as sentiment analysis or
        entity extraction) is derived from the final representation of the data.
        
        The method includes error handling for the annotation process to prevent
        a single failing annotator from crashing the entire pipeline. If an
        exception occurs during annotation, the error message is recorded in the
        annotation slot instead of the result.

        Args:
            content: The primary text content of the payload (e.g., summary, translation).
            context: The execution context containing the registry of annotators.
            content_raw: The original source text. If not provided, or if identical 
                to content, it will be stored as None.

        Returns:
            A fully initialized ChunkPayload instance with populated annotations.
        """
        if content_raw == content:
            content_raw = None

        payload = ChunkPayload(
            content=content,
            content_raw=content_raw,
            annotations={}
        )
        
        if context and context.annotators:
            annotator_context = ExecutionContext() 
            
            for alias, annotator in context.annotators.items():
                try:
                    annotation_result = annotator.execute(content, annotator_context)
                    payload.annotations[alias] = annotation_result
                except Exception as e:
                    payload.annotations[alias] = {"error": str(e)}
            
        return payload

    def _apply_annotators_to_list(
        self, 
        items: List[str], 
        context: ExecutionContext
    ) -> List[ChunkPayload]:
        """
        Helper utility to convert a simple list of strings into a list of
        annotated ChunkPayloads.

        This method is primarily used by simpler strategies or Mock implementations
        that map input strings directly to output payloads 1-to-1 without complex
        structural logic. It delegates the creation of each individual payload
        to the _apply_annotators_to_payload factory method.

        Args:
            items: A list of text strings to be converted.
            context: The execution context containing annotators.

        Returns:
            A list of fully annotated ChunkPayload objects.
        """
        return [self._apply_annotators_to_payload(item, context, content_raw=item) for item in items]