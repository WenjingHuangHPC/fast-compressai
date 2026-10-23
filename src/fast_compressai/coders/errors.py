"""Errors raised by coder backends and bitstream validation."""


class CoderError(RuntimeError):
    """Base class for coder failures."""


class CoderInputError(CoderError, ValueError):
    """Raised when symbols, indexes, or CDF metadata is invalid."""


class CoderStateError(CoderError):
    """Raised when a coder session is used in an invalid state."""


class BitstreamError(CoderError, ValueError):
    """Raised for malformed or incompatible bitstreams."""


class CoderDependencyError(CoderError, ImportError):
    """Raised when an optional coder dependency is unavailable."""
