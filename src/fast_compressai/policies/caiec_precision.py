"""CAIEC-style operator sensitivity ranking and precision recovery."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from ..profiling.schema import ProfileRecord

SELECTOR_VERSION = "caiec-operator-v1"
RD_COST_THRESHOLD = 1.0
THR_GAIN_THRESHOLD = 1.5


@dataclass(frozen=True)
class Sensitivity:
    instance_id: str
    rd_cost: float
    thr_gain: float
    available_precisions: tuple[str, ...]
    unavailable_reasons: tuple[str, ...] = ()
    candidate_key: object | None = None

    @property
    def category(self) -> str:
        """Classify one-at-a-time sensitivity, not the final RD budget."""
        if self.rd_cost > RD_COST_THRESHOLD:
            return "high_cost"
        if self.thr_gain > THR_GAIN_THRESHOLD:
            return "high_gain"
        return "low_impact"


def rd_cost(bpp: float, bpp_ref: float, psnr: float, psnr_ref: float) -> float:
    if min(bpp, bpp_ref) <= 0:
        raise ValueError("bpp and reference bpp must be positive")
    return math.log10(bpp / bpp_ref) + (psnr_ref - psnr)


def thr_gain(reference_latency: float, candidate_latency: float, eligible_count: int) -> float:
    if reference_latency <= 0 or candidate_latency < 0 or eligible_count < 1:
        raise ValueError("invalid latency or eligible_count")
    return 1.0 + eligible_count * (reference_latency - candidate_latency) / reference_latency


def make_sensitivity(
    instance_id: str,
    *,
    reference: ProfileRecord,
    perturbation: ProfileRecord,
    reference_bpp: float,
    reference_psnr: float,
    eligible_count: int,
    available_precisions: Sequence[str],
    unavailable_reasons: Sequence[str] = (),
) -> Sensitivity:
    if reference.latency_ms is None or perturbation.latency_ms is None:
        raise ValueError(f"missing latency profile for {instance_id}")
    if perturbation.bpp is None or perturbation.psnr_db is None:
        raise ValueError(f"missing complete RD measurements for {instance_id}")
    return Sensitivity(
        instance_id=instance_id,
        rd_cost=rd_cost(perturbation.bpp, reference_bpp, perturbation.psnr_db, reference_psnr),
        thr_gain=thr_gain(reference.latency_ms, perturbation.latency_ms, eligible_count),
        available_precisions=tuple(sorted(set(available_precisions))),
        unavailable_reasons=tuple(unavailable_reasons),
    )


def initial_assignments(sensitivities: Sequence[Sensitivity]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in sensitivities:
        if item.category == "high_cost":
            result[item.instance_id] = "fp16" if "fp16" in item.available_precisions else "fp32"
        elif "fp8" in item.available_precisions:
            result[item.instance_id] = "fp8"
        elif "fp16" in item.available_precisions:
            # An unavailable FP8 probe is never fabricated; use the closest
            # supported conservative candidate and retain the reason.
            result[item.instance_id] = "fp16"
        else:
            result[item.instance_id] = "fp32"
    return result


def recovery_order(sensitivities: Sequence[Sensitivity]) -> tuple[str, ...]:
    groups = {"low_impact": [], "high_gain": [], "high_cost": []}
    for item in sensitivities:
        groups[item.category].append(item)
    ordered: list[str] = []
    for category in ("low_impact", "high_gain", "high_cost"):
        group = sorted(groups[category], key=lambda item: (-item.rd_cost, item.instance_id))
        ordered.extend(item.instance_id for item in group)
    return tuple(ordered)


def recover_to_budget(
    initial: Mapping[str, str],
    sensitivities: Sequence[Sensitivity],
    validate_full_configuration: Callable[[Mapping[str, str]], float],
    epsilon: float,
) -> tuple[dict[str, str], float, tuple[str, ...], tuple[dict[str, object], ...], bool]:
    """Promote lower precision candidates and remeasure each full configuration."""

    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    assignment = dict(initial)
    cost = float(validate_full_configuration(assignment))
    changed: list[str] = []
    steps: list[dict[str, object]] = [{
        "step": 0,
        "changed_instance_id": None,
        "from_precision": None,
        "to_precision": None,
        "validated_rd_cost": cost,
        "budget_met": cost <= epsilon,
    }]
    if cost <= epsilon:
        return assignment, cost, tuple(changed), tuple(steps), True

    by_id = {item.instance_id: item for item in sensitivities}
    for instance_id in recovery_order(sensitivities):
        current = assignment.get(instance_id, "fp32")
        available = by_id[instance_id].available_precisions
        if current == "fp8" and "fp16" in available:
            next_precision = "fp16"
        elif current in {"fp8", "fp16"}:
            next_precision = "fp32"
        else:
            continue
        assignment[instance_id] = next_precision
        changed.append(instance_id)
        cost = float(validate_full_configuration(assignment))
        steps.append({
            "step": len(steps),
            "changed_instance_id": instance_id,
            "from_precision": current,
            "to_precision": next_precision,
            "validated_rd_cost": cost,
            "budget_met": cost <= epsilon,
        })
        if cost <= epsilon:
            return assignment, cost, tuple(changed), tuple(steps), True
    return assignment, cost, tuple(changed), tuple(steps), False
