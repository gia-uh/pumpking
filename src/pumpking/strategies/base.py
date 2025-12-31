from typing import Any, List, Optional
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
    
    SUPPORTED_INPUTS: List[Any] = [str, List[str]]
    PRODUCED_OUTPUT: Any = List[ChunkPayload]

    def execute(self, data: Any, context: ExecutionContext) -> Any:
        """
        Abstract method that must be implemented by all concrete strategies.

        This method defines the specific processing logic of the strategy (e.g.,
        splitting text by paragraphs, clustering by entities, or summarizing).
        It receives raw data and an execution context, and it must return the
        transformed data defined in PRODUCED_OUTPUT.

        Args:
            data: The input data to be processed.
            context: The shared execution context containing configuration and annotators.

        Returns:
            The processed result, typically a list of ChunkPayloads.

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
            content_raw: The original source text. If not provided, it defaults to 'content'.

        Returns:
            A fully initialized ChunkPayload instance with populated annotations.
        """
        payload = ChunkPayload(
            content=content,
            content_raw=content_raw or content,
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
        return [self._apply_annotators_to_payload(item, context) for item in items]