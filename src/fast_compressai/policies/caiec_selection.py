"""CAIEC operator-level selection for Balle CUDA models."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Sequence

import torch
from torch import nn

from ..configs.precision import PrecisionAssignment, PrecisionConfig, fingerprint_json, model_fingerprint
from ..operators.catalog import OperatorCatalog
from ..operators.candidates import CandidateKey
from ..profiling.rd import CalibrationData, FactorizedRDMeasurement, measure_factorized_warpdiv_rd
from ..profiling.schema import ProfileRecord
from .caiec_precision import SELECTOR_VERSION, Sensitivity, initial_assignments, recover_to_budget


@dataclass(frozen=True)
class SelectionResult:
    model: nn.Module
    config: PrecisionConfig
    rd_cost: float
    profile_cost_ms: float
    applications: tuple


def normalize_calibration(data, device: torch.device) -> list[torch.Tensor]:
    if isinstance(data, torch.Tensor):
        values = [data[index:index + 1] for index in range(data.shape[0])] if data.ndim == 4 else [data]
    else:
        values = list(data)
    if not values or any(not isinstance(value, torch.Tensor) for value in values):
        raise ValueError("calibration_data must contain Tensor samples")
    return [value.to(device) for value in values]


def select_balle_precision(
    model: nn.Module,
    catalog: OperatorCatalog,
    calibration: CalibrationData,
    *,
    epsilon: float,
    p_ch: int,
    model_name: str,
    device_capability: tuple[int, int],
    coder: str = "warp_div",
    warmup: int = 0,
    repetitions: int = 1,
) -> SelectionResult:
    started = time.perf_counter()
    inventory = catalog.inventory(model)
    eligible = [item for item in inventory if item.logical_op in {"conv2d", "conv_transpose2d", "gdn", "igdn"}]
    if not eligible:
        raise ValueError("no eligible Balle floating operator instances were found")

    # EntropyBottleneck is intentionally excluded: its discrete coder path is
    # precision-locked and is not required to have an FP16 candidate.
    all_fp16: dict[str, CandidateKey] = {}
    available: dict[str, tuple[str, ...]] = {}
    unavailable_candidates: dict[str, tuple[str, ...]] = {}
    fp8_candidates_by_id: dict[str, tuple[CandidateKey, ...]] = {}
    for instance in eligible:
        entry = catalog.get(instance.logical_op)
        keys = [candidate.key for candidate in entry.candidates]
        available[instance.instance_id] = tuple(sorted({key.precision for key in keys}))
        precision_reasons: list[str] = []
        for key in keys:
            try:
                candidate = catalog.candidate_by_key(key)
                original = dict(model.named_modules(remove_duplicate=False))[instance.module_path]
                supported, reason = candidate.capability(original, {**dict(instance.config), "training": False})
                if not supported:
                    precision_reasons.append(f"{key.precision}/{key.backend}: {reason or 'capability rejected'}")
            except Exception as exc:
                precision_reasons.append(f"{key.precision}/{key.backend}: {exc}")
        if "fp8" not in available[instance.instance_id]:
            precision_reasons.append("fp8: no implementation registered")
        unavailable_candidates[instance.instance_id] = tuple(precision_reasons)
        fp16 = next((key for key in keys if key.precision == "fp16"), None)
        if fp16 is None:
            raise ValueError(f"FP16 reference unavailable for eligible operator {instance.module_path}")
        all_fp16[instance.instance_id] = fp16
        supported_fp8 = []
        original = dict(model.named_modules(remove_duplicate=False))[instance.module_path]
        context = {**dict(instance.config), "training": False}
        for key in keys:
            if key.precision != "fp8":
                continue
            candidate = catalog.candidate_by_key(key)
            supported, _ = candidate.capability(original, context)
            if supported:
                supported_fp8.append(key)
        fp8_candidates_by_id[instance.instance_id] = tuple(
            sorted(supported_fp8, key=lambda item: item.stable_id)
        )

    reference_model, _ = catalog.apply(model, all_fp16, backend=None)
    reference_raw = measure_factorized_warpdiv_rd(
        reference_model, calibration, p_ch=p_ch, model_name=model_name,
        reference_bpp=1.0, reference_psnr_db=0.0,
        warmup=warmup, repetitions=repetitions,
    )
    # Use the all-FP16 measurement as its own RD reference (zero by definition).
    reference = _with_reference_cost(reference_raw, reference_raw.bpp, reference_raw.psnr_db)
    sensitivity: list[Sensitivity] = []
    sensitivity_probe_records: list[ProfileRecord] = []
    selection_records: list[dict] = []
    selected_fp8: dict[str, CandidateKey] = {}
    for instance in eligible:
        fp8_candidates = fp8_candidates_by_id[instance.instance_id]
        if not fp8_candidates:
            # Emit an explicit unavailable probe record for every eligible
            # instance; CAIEC sensitivity coverage must not silently omit ops.
            unavailable_reason = "; ".join(unavailable_candidates[instance.instance_id])
            unavailable_record = ProfileRecord(
                kind="operator", target=instance.instance_id,
                candidate="fp8/unavailable", context={
                    "logical_op": instance.logical_op,
                    "module_path": instance.module_path,
                }, provenance="measured", unavailable_reason=unavailable_reason,
            )
            sensitivity_probe_records.append(unavailable_record)
            selection_records.append({
                "instance_id": instance.instance_id,
                "module_path": instance.module_path,
                "logical_op": instance.logical_op,
                "candidate_key": None,
                "status": "unavailable",
                "probe_rd_cost": None,
                "probe_thr_gain": None,
                "category": "unavailable",
                "unavailable_reason": unavailable_reason,
            })
            continue
        best_key = None
        best_probe = None
        for fp8 in fp8_candidates:
            probe_map = dict(all_fp16)
            probe_map[instance.instance_id] = fp8
            probe_model, _ = catalog.apply(model, probe_map, backend=None)
            probe = measure_factorized_warpdiv_rd(
                probe_model, calibration, p_ch=p_ch, model_name=model_name,
                reference_bpp=reference.bpp, reference_psnr_db=reference.psnr_db,
                warmup=warmup, repetitions=repetitions,
            )
            record = replace(
                probe.record,
                candidate=fp8.stable_id,
                context={**dict(probe.record.context), "candidate_key": fp8.to_dict()},
            )
            sensitivity_probe_records.append(record)
            selection_records.append({
                "instance_id": instance.instance_id,
                "module_path": instance.module_path,
                "logical_op": instance.logical_op,
                "candidate_key": fp8.to_dict(),
                "status": "measured",
                "probe_rd_cost": probe.rd_cost,
                "probe_thr_gain": 1.0 + len(eligible) * (
                    reference.forward_latency_ms - probe.forward_latency_ms
                ) / max(reference.forward_latency_ms, 1e-12),
                "category": Sensitivity(
                    instance_id=instance.instance_id, rd_cost=probe.rd_cost,
                    thr_gain=1.0 + len(eligible) * (
                        reference.forward_latency_ms - probe.forward_latency_ms
                    ) / max(reference.forward_latency_ms, 1e-12),
                    available_precisions=available[instance.instance_id],
                ).category,
                "forward_latency_ms": probe.forward_latency_ms,
            })
            if best_probe is None or (probe.rd_cost, probe.forward_latency_ms, fp8.stable_id) < (
                best_probe.rd_cost, best_probe.forward_latency_ms, best_key.stable_id
            ):
                best_key, best_probe = fp8, probe
        selected_fp8[instance.instance_id] = best_key
        sensitivity.append(Sensitivity(
            instance_id=instance.instance_id,
            rd_cost=best_probe.rd_cost,
            thr_gain=1.0 + len(eligible) * (
                reference.forward_latency_ms - best_probe.forward_latency_ms
            ) / max(reference.forward_latency_ms, 1e-12),
            available_precisions=available[instance.instance_id],
        ))

    initial = {key: value.precision for key, value in all_fp16.items()}
    for item in sensitivity:
        if item.category != "high_cost" and "fp8" in item.available_precisions:
            initial[item.instance_id] = "fp8"

    def make_map(precision_map):
        result = {}
        for item in eligible:
            precision = precision_map[item.instance_id]
            if precision == "fp8":
                result[item.instance_id] = selected_fp8[item.instance_id]
            elif precision == "fp16":
                result[item.instance_id] = all_fp16[item.instance_id]
            else:
                result[item.instance_id] = next(
                    candidate.key for candidate in catalog.get(item.logical_op).candidates
                    if candidate.key.precision == precision
                )
        return result

    cache: dict[tuple[tuple[str, str], ...], FactorizedRDMeasurement] = {}

    def validate(precision_map):
        key = tuple(sorted(precision_map.items()))
        if key not in cache:
            configured, _ = catalog.apply(model, make_map(precision_map), backend=None)
            cache[key] = measure_factorized_warpdiv_rd(
                configured, calibration, p_ch=p_ch, model_name=model_name,
                reference_bpp=reference.bpp, reference_psnr_db=reference.psnr_db,
                warmup=warmup, repetitions=repetitions,
            )
        return cache[key].rd_cost

    final_precisions, _, recovered, recovery_steps, met = recover_to_budget(initial, sensitivity, validate, epsilon)
    final_map = make_map(final_precisions)
    final_model, applications = catalog.apply(model, final_map, backend=None)
    final_rd = validate(final_precisions)
    final_measurement = cache[tuple(sorted(final_precisions.items()))]
    assignments = []
    sensitivity_by_id = {item.instance_id: item for item in sensitivity}
    instances = {item.instance_id: item for item in eligible}
    for instance_id in sorted(final_map):
        instance = instances[instance_id]
        key = final_map[instance_id]
        metric = sensitivity_by_id.get(instance_id)
        assignments.append(PrecisionAssignment(
            instance_id=instance_id,
            module_path=instance.module_path,
            logical_op=instance.logical_op,
            candidate_key=key.to_dict(),
            precision=key.precision,
            backend=key.backend,
            category=metric.category if metric else "unsupported_fp8",
            probe_rd_cost=metric.rd_cost if metric else 0.0,
            probe_thr_gain=metric.thr_gain if metric else 0.0,
            fallback_reason=None if metric else "no FP8 candidate for this operator",
            selection_reason=(
                "recovered_to_fp16_or_fp32" if metric and key.precision != "fp8"
                else "selected_fp8_candidate" if metric else "no_fp8_candidate",
            ),
        ))
    config = PrecisionConfig(
        model_name=model_name,
        model_fingerprint=model_fingerprint(model),
        dataset_name=calibration.dataset_name,
        dataset_fingerprint=calibration.dataset_fingerprint,
        dataset_sample_count=len(calibration.samples),
        device_type="cuda",
        device_capability=device_capability,
        coder=coder,
        p_ch=p_ch,
        epsilon=epsilon,
        rd_cost_formula="log10(bpp/bpp_ref)+(psnr_ref-psnr)",
        selector_version=SELECTOR_VERSION,
        catalog_fingerprint=catalog.catalog_fingerprint(),
        assignments=tuple(assignments),
        probe_records=tuple(selection_records),
        validated_rd_cost=final_rd,
        budget_met=bool(met and final_rd <= epsilon),
        recovery_order=recovered,
        recovery_steps=tuple(recovery_steps),
        records_path=None,
        recovery_path=None,
        profile_cost_ms=(time.perf_counter() - started) * 1000.0,
        created_utc=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"),
    )
    return SelectionResult(final_model, config, final_rd, config.profile_cost_ms, applications)


def _with_reference_cost(measurement: FactorizedRDMeasurement, bpp_ref: float, psnr_ref: float):
    # The reference's RD Cost against itself is exactly zero. Use dataclasses.replace
    # so the stored record has the same semantics as perturbation records.
    from dataclasses import replace
    record = replace(measurement.record, rd_cost=0.0, bpp=bpp_ref, psnr_db=psnr_ref)
    return replace(measurement, rd_cost=0.0, record=record)
