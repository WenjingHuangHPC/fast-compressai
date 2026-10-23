"""Operator registration and explicit dispatch APIs."""

from .errors import OperatorSelectionError, RegistryError
from .candidates import CandidateKey, OperatorCandidate, OperatorInstance, OperatorSpec
from .catalog import CandidateApplication, CatalogEntry, OperatorCatalog, build_balle_catalog
from .interfaces import (
    OperatorContext,
    OperatorDecision,
    OperatorDefinition,
    OperatorImplementation,
)
from .registry import (
    OperatorRegistry,
    default_registry,
    dispatch,
    get_operator,
    register_operator,
    unregister_operator,
)

__all__ = [
    "CandidateApplication",
    "CandidateKey",
    "CatalogEntry",
    "OperatorCandidate",
    "OperatorCatalog",
    "OperatorInstance",
    "OperatorSpec",
    "build_balle_catalog",
    "OperatorContext",
    "OperatorDecision",
    "OperatorDefinition",
    "OperatorImplementation",
    "OperatorRegistry",
    "OperatorSelectionError",
    "RegistryError",
    "default_registry",
    "dispatch",
    "get_operator",
    "register_operator",
    "unregister_operator",
]
