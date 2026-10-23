"""Common runtime errors for Fast-CompressAI."""


class UnsupportedOperationError(NotImplementedError):
    """Raised when a model path is not available in the current adapter."""
