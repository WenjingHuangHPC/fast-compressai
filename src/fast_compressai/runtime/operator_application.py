"""Runtime construction/application helpers for per-instance candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from torch import nn

from ..operators.catalog import CandidateApplication, OperatorCatalog


@dataclass(frozen=True)
class AppliedOperatorPlan:
    model: nn.Module
    applications: tuple[CandidateApplication, ...]
    inventory: tuple


def apply_operator_plan(
    model: nn.Module,
    catalog: OperatorCatalog,
    assignments: Mapping[str, str],
    *,
    backend: str = "pytorch-cudnn",
) -> AppliedOperatorPlan:
    runtime_model, applications = catalog.apply(model, assignments, backend=backend)
    inventory = catalog.inventory(runtime_model)
    return AppliedOperatorPlan(runtime_model, applications, inventory)
