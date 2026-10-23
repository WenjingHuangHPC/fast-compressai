"""Errors raised by the operator registry."""


class RegistryError(ValueError):
    """Raised for invalid registry operations or unknown operators."""


class OperatorSelectionError(RegistryError):
    """Raised when an implementation cannot be selected safely."""
