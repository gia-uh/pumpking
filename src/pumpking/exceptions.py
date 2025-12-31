class PumpkingError(Exception):
    """Base exception for all Pumpking related errors."""
    pass

class PipelineConfigurationError(PumpkingError):
    """
    Raised when there are issues with the pipeline structure, 
    such as type mismatches, alias collisions, or invalid topologies.
    """
    pass

class ValidationError(PumpkingError):
    """Raised when graph or model validation fails."""
    pass

class IntegrityError(ValidationError):
    """Raised when graph structural integrity is compromised."""
    pass

class TextCoverageError(ValidationError):
    """Raised when text coverage between nodes is inconsistent."""
    pass