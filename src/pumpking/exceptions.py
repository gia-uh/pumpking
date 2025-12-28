class PumpkingError(Exception):
    """Base exception for all Pumpking errors."""
    pass


class PipelineConfigurationError(PumpkingError):
    """Raised when the pipeline configuration is invalid (e.g. type mismatch)."""
    pass