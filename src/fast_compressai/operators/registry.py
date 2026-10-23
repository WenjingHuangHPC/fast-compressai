"""Explicit operator registration, selection, and dispatch."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .errors import OperatorSelectionError, RegistryError
from .interfaces import (
    Capability,
    Implementation,
    OperatorContext,
    OperatorDecision,
    OperatorDefinition,
    OperatorImplementation,
)


class OperatorRegistry:
    """An isolated registry of callable operator implementations."""

    def __init__(self) -> None:
        self._definitions: dict[str, OperatorDefinition] = {}
        self._type_keys: dict[type, str] = {}

    def register(
        self,
        key: str,
        reference: Implementation,
        implementations: Mapping[str, Implementation | tuple[Implementation, Capability | None]] | None = None,
        *,
        capabilities: Mapping[str, Capability] | None = None,
        fallback: str = "fp32",
        op_type: type | None = None,
    ) -> OperatorDefinition:
        """Register an operator without silently replacing an existing key."""

        if key in self._definitions:
            raise RegistryError(f"operator {key!r} is already registered")
        if op_type is not None and op_type in self._type_keys:
            raise RegistryError(f"operator type {op_type.__name__!r} is already registered")
        capabilities = {} if capabilities is None else dict(capabilities)
        implementations = {} if implementations is None else dict(implementations)
        unknown_capabilities = set(capabilities) - set(implementations)
        if unknown_capabilities:
            names = ", ".join(sorted(map(str, unknown_capabilities)))
            raise RegistryError(f"capability provided for unregistered precision(s): {names}")

        records: list[OperatorImplementation] = []
        for precision, value in implementations.items():
            if not isinstance(precision, str) or not precision.strip():
                raise RegistryError("implementation precision must be a non-empty string")
            capability = capabilities.get(precision)
            function = value
            if isinstance(value, tuple):
                if len(value) != 2:
                    raise RegistryError(
                        f"implementation tuple for {precision!r} must be (function, capability)"
                    )
                function, tuple_capability = value
                if capability is not None and tuple_capability is not None:
                    raise RegistryError(f"capability for {precision!r} was specified twice")
                capability = tuple_capability if capability is None else capability
            records.append(OperatorImplementation(str(precision), function, capability))

        definition = OperatorDefinition(key, reference, tuple(records), fallback, op_type)
        self._definitions[key] = definition
        if op_type is not None:
            self._type_keys[op_type] = key
        return definition

    def unregister(self, key_or_type: str | type) -> OperatorDefinition:
        key = self._canonical_key(key_or_type)
        definition = self._definitions.pop(key)
        if definition.op_type is not None:
            self._type_keys.pop(definition.op_type, None)
        return definition

    def _canonical_key(self, key_or_type: str | type) -> str:
        if isinstance(key_or_type, str):
            return key_or_type
        if isinstance(key_or_type, type) and key_or_type in self._type_keys:
            return self._type_keys[key_or_type]
        if isinstance(key_or_type, type):
            raise RegistryError(f"operator type {key_or_type.__name__!r} is not registered")
        raise RegistryError("operator identity must be a string key or type")

    def get(self, key_or_type: str | type) -> OperatorDefinition:
        key = self._canonical_key(key_or_type)
        try:
            return self._definitions[key]
        except KeyError as exc:
            raise RegistryError(f"operator {key!r} is not registered") from exc

    def snapshot(self) -> tuple[OperatorDefinition, ...]:
        """Return an immutable snapshot of the current registrations."""

        return tuple(self._definitions.values())

    def clear(self) -> None:
        self._definitions.clear()
        self._type_keys.clear()

    def select(
        self,
        key_or_type: str | type,
        *,
        precision: str | None = None,
        context: OperatorContext | Mapping[str, Any] | None = None,
        allow_fallback: bool = True,
    ) -> tuple[OperatorDecision, Implementation]:
        definition = self.get(key_or_type)
        context = OperatorContext.from_value(context)
        requested = precision
        if requested is not None and (not isinstance(requested, str) or not requested.strip()):
            raise OperatorSelectionError("requested precision must be a non-empty string or None")
        if requested is not None and not isinstance(requested, str):
            raise OperatorSelectionError("requested precision must be a non-empty string or None")
        if requested in (None, "fp32"):
            return (
                OperatorDecision(
                    definition.key, requested, "fp32", "reference", False, False,
                    "reference implementation selected",
                ),
                definition.reference,
            )

        implementation = definition.implementation_map.get(requested)
        if implementation is None:
            return self._fallback(
                definition, requested, context, allow_fallback,
                f"precision {requested!r} is not registered",
            )

        checked_context = context.with_precision(requested)
        if implementation.capability is not None:
            try:
                supported = implementation.capability(checked_context)
            except Exception as exc:
                return self._fallback(
                    definition, requested, context, allow_fallback,
                    f"capability evaluation failed: {exc}",
                )
            if not isinstance(supported, bool):
                return self._fallback(
                    definition, requested, context, allow_fallback,
                    "capability predicate did not return bool",
                )
            if not supported:
                return self._fallback(
                    definition, requested, context, allow_fallback,
                    f"capability rejected context {checked_context}",
                )

        return (
            OperatorDecision(
                definition.key, requested, requested, requested, True, False,
                "requested implementation selected",
            ),
            implementation.function,
        )

    def _fallback(
        self,
        definition: OperatorDefinition,
        requested: str,
        context: OperatorContext,
        allow_fallback: bool,
        reason: str,
    ) -> tuple[OperatorDecision, Implementation]:
        if not allow_fallback:
            raise OperatorSelectionError(
                f"cannot select operator {definition.key!r} at precision {requested!r}; "
                f"reference fallback is disabled ({reason}); context={context}"
            )
        return (
            OperatorDecision(
                definition.key, requested, "fp32", "reference", False, True,
                f"fell back to reference: {reason}",
            ),
            definition.reference,
        )

    def dispatch(
        self,
        key_or_type: str | type,
        *args: Any,
        precision: str | None = None,
        context: OperatorContext | Mapping[str, Any] | None = None,
        allow_fallback: bool = True,
        **kwargs: Any,
    ) -> Any:
        if context is None:
            context = OperatorContext.from_inputs(*args, **kwargs)
        _, implementation = self.select(
            key_or_type,
            precision=precision,
            context=context,
            allow_fallback=allow_fallback,
        )
        return implementation(*args, **kwargs)


# The default registry is intentionally process-local. Applications and tests can
# use OperatorRegistry() when isolation from global registrations is required.
default_registry = OperatorRegistry()


def register_operator(*args: Any, **kwargs: Any) -> OperatorDefinition:
    return default_registry.register(*args, **kwargs)


def unregister_operator(key_or_type: str | type) -> OperatorDefinition:
    return default_registry.unregister(key_or_type)


def get_operator(key_or_type: str | type) -> OperatorDefinition:
    return default_registry.get(key_or_type)


def dispatch(key_or_type: str | type, *args: Any, **kwargs: Any) -> Any:
    return default_registry.dispatch(key_or_type, *args, **kwargs)
