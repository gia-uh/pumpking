class PumpkingError(Exception):
    """
    Base exception class for all errors originating within the Pumpking architecture.

    This class serves as the root of the custom exception hierarchy. Catching this
    exception allows calling code to handle any Pumpking-specific failure gracefully
    without inadvertently catching unrelated system exceptions or standard library errors.
    """
    pass

class PipelineConfigurationError(PumpkingError):
    """
    Exception raised when the pipeline configuration is invalid or inconsistent.

    This error is typically thrown during the initialization or assembly phase of the
    pipeline. Common causes include:
    - Type mismatches between connected strategies.
    - Duplicate aliases defined for annotators or steps.
    - Invalid topologies, such as cycles in the execution graph or disconnected components.
    """
    pass

class ValidationError(PumpkingError):
    """
    Base exception for failures related to data validity or graph correctness.

    This exception indicates that a model, payload, or graph structure has violated
    the defined constraints or business rules. It serves as a parent category for
    more specific validation issues like integrity breaches or coverage gaps.
    """
    pass

class IntegrityError(ValidationError):
    """
    Exception raised when the structural integrity of the execution graph is compromised.

    This error occurs when the relationships between nodes are invalid or broken.
    Examples include:
    - A child node referencing a non-existent parent ID.
    - Orphaned nodes that are not reachable from the document root.
    - Inconsistent lineage data where a child does not properly map back to its source.
    """
    pass

class TextCoverageError(ValidationError):
    """
    Exception raised when there is an inconsistency in text coverage validation.

    This specific validation error is triggered when the text content of child nodes
    does not accurately reflect or cover the content of their parent node. This is
    critical for ensuring that no information is lost or hallucinated during
    chunking or transformation processes.
    """
    pass