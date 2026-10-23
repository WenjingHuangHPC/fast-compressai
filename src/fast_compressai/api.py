"""User-facing compile and optimize entry points."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .adapters import CompressAIAdapter
from .config import AccelerationConfig, CompileConfig, ConfigurationError
from .configs import PrecisionAssignment, PrecisionConfig, PrecisionConfigError, fingerprint_json, model_fingerprint
from .policies.caiec_selection import normalize_calibration, select_balle_precision
from .profiling.rd import CalibrationData, calibration_fingerprint
from .policies.caiec_selection import select_balle_precision
from .coders.factorized import FactorizedWarpDivCodec
from .errors import UnsupportedOperationError
from .operators import (
    OperatorContext,
    OperatorDecision,
    OperatorRegistry,
    build_balle_catalog,
    default_registry,
)
from .policies import PrecisionPolicy, PrecisionRequest
from .profiling.operator import make_bf16_conv_candidate, profile_forward_pair


@dataclass(frozen=True)
class CompileReport:
    """Explain compatibility, policy evaluation, and operator selections."""

    backend: str
    accelerated: bool
    available_paths: tuple[str, ...]
    missing_paths: tuple[str, ...]
    fallback_used: bool
    message: str
    operator_decisions: tuple[OperatorDecision, ...] = ()
    operator_fallback_used: bool = False
    candidate_accelerated: bool = False
    runtime_applied: bool = False
    framework: str | None = None
    framework_version: str | None = None
    model_class: str | None = None
    model_module: str | None = None
    contract_family: str = "generic"
    update_available: bool = False
    update_called: bool = False
    entropy_models: tuple[str, ...] = ()
    signature_level: str = "unknown"
    compress_signature: str | None = None
    decompress_signature: str | None = None
    validation_level: str = "unvalidated"
    precision_policy_type: str = "fp32"
    precision_policy_schema_version: int = 1
    precision_policy_serialized: Mapping[str, Any] | None = None
    precision_map: Mapping[str, str] = ()
    precision_sources: Mapping[str, str] = ()
    requested_precisions: Mapping[str, str] = ()
    selected_precisions: Mapping[str, str] = ()
    precision_fallback_used: bool = False
    precision_runtime_applied: bool = False
    precision_evaluation_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["operator_decisions"] = [
            decision.to_dict() for decision in self.operator_decisions
        ]
        for field in (
            "precision_map",
            "precision_sources",
            "requested_precisions",
            "selected_precisions",
        ):
            result[field] = dict(getattr(self, field))
        if self.precision_policy_serialized is not None:
            result["precision_policy_serialized"] = dict(self.precision_policy_serialized)
        return result


@dataclass(frozen=True)
class AccelerationReport:
    """Truthful status for the staged automatic acceleration workflow."""

    status: str
    model_class: str
    adapter_family: str
    p_ch: int | None
    quality_budget: float
    quality_metric: str
    profiling_status: str
    precision_selection_status: str
    selected_candidate: str
    psnr_drop_db: float | None
    reference_latency_ms: float | None
    candidate_latency_ms: float | None
    coder_status: str
    p_ch_applied: bool
    runtime_status: str
    speedup: float | None
    message: str
    rd_cost: float | None = None
    budget_met: bool | None = None
    precision_config_path: str | None = None
    sensitivity_probe_count: int = 0
    unavailable_probe_count: int = 0
    recovered_instance_count: int = 0

    @property
    def accelerated(self) -> bool:
        return self.runtime_status == "applied" and self.speedup is not None and self.speedup > 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _save_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False), encoding="utf-8")


def accelerate(
    model: nn.Module,
    P: int | None = None,
    budget: float | None = None,
    *,
    config: AccelerationConfig | Mapping[str, Any] | None = None,
    device: str | None = None,
    sample: Any | None = None,
    coder: str | None = None,
    calibration_data: Any | None = None,
    dataset_name: str | None = None,
    precision_config_path: str | None = None,
    precision_config_output_dir: str | Path | None = None,
) -> "FastModel":
    """Prepare the automatic acceleration workflow around a model.

    ``P`` is GA-ANS channel granularity. The default ``warp_div`` coder
    connects the local CUDA backend to Balle FactorizedPrior compression;
    inference remains FP32. Select ``coder='rans'`` to retain the model's
    original RANS entropy path (and the optional CPU BF16-forward experiment).
    """

    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an instance of torch.nn.Module, got {type(model).__name__}")
    acceleration_config = AccelerationConfig.from_value(
        config, P=P, budget=budget, device=device, coder=coder,
        precision_config_path=precision_config_path,
    )
    selected_coder = acceleration_config.coder
    compile_config = CompileConfig(device=acceleration_config.device)
    codec = None
    precision_selection_result = None
    final_rd_cost = None
    final_budget_met = None
    config_path_used = precision_config_path
    profile_probe_count = 0
    unavailable_probe_count = 0
    recovered_count = 0
    if precision_config_path is not None and calibration_data is not None:
        raise ConfigurationError("do not combine precision_config_path with calibration_data")
    precision_config = None
    if selected_coder == "warp_div":
        try:
            from compressai.models import FactorizedPrior
        except ImportError as exc:
            raise ConfigurationError("warp_div model integration requires CompressAI") from exc
        if type(model) is not FactorizedPrior:
            raise ConfigurationError("warp_div currently supports only CompressAI FactorizedPrior")
        if sample is not None:
            raise ConfigurationError("warp_div precision selection uses calibration_data; legacy sample profiling is unavailable")
        if not torch.cuda.is_available():
            raise ConfigurationError("warp_div requires CUDA")
        target = torch.device(acceleration_config.device or "cuda")
        if target.type != "cuda":
            raise ConfigurationError("warp_div requires device='cuda'")
        runtime_model = copy.deepcopy(model).to(target).eval()
        catalog = build_balle_catalog()
        p_ch = acceleration_config.p_ch if acceleration_config.p_ch is not None else 1
        if precision_config_path is not None:
            precision_config = PrecisionConfig.load(precision_config_path)
            actual_model_fingerprint = model_fingerprint(model)
            if precision_config.model_fingerprint != actual_model_fingerprint:
                raise ConfigurationError("precision config model fingerprint does not match the input model")
            if precision_config.coder != selected_coder or precision_config.p_ch != (acceleration_config.p_ch or 1):
                raise ConfigurationError("precision config coder/P_ch does not match the request")
            if precision_config.catalog_fingerprint != catalog.catalog_fingerprint():
                raise ConfigurationError("precision config catalog fingerprint does not match current candidates")
            dataset_fingerprint = calibration_data.dataset_fingerprint if calibration_data is not None else None
            if dataset_fingerprint is not None and precision_config.dataset_fingerprint != dataset_fingerprint:
                raise ConfigurationError("precision config dataset fingerprint does not match calibration data")
            precision_config.validate_compatibility(
                model_fingerprint=actual_model_fingerprint,
                dataset_fingerprint=precision_config.dataset_fingerprint if dataset_fingerprint is None else dataset_fingerprint,
                device_type="cuda",
                device_capability=torch.cuda.get_device_capability(target),
                coder=selected_coder,
                p_ch=acceleration_config.p_ch or 1,
                catalog_fingerprint=catalog.catalog_fingerprint(),
            )
            final_rd_cost = precision_config.validated_rd_cost
            final_budget_met = precision_config.budget_met
            unavailable_probe_count = sum(
                1 for record in precision_config.probe_records
                if record.get("unavailable_reason") is not None
            )
            profile_probe_count = len(precision_config.probe_records)
            recovered_count = len(precision_config.recovery_order)
            if precision_config.dataset_name != (dataset_name or precision_config.dataset_name):
                raise ConfigurationError("precision config dataset name does not match the request")
            from .operators import CandidateKey
            assignments = {assignment.instance_id: CandidateKey.from_dict(assignment.candidate_key) for assignment in precision_config.assignments}
            expected_instances = {instance.instance_id for instance in catalog.inventory(runtime_model)
                                  if instance.logical_op in {"conv2d", "conv_transpose2d", "gdn", "igdn"}}
            if set(assignments) != expected_instances:
                raise ConfigurationError("precision config operator inventory does not match the current model")
            try:
                runtime_model, applications = catalog.apply(runtime_model, assignments, backend=None)
            except Exception as exc:
                raise ConfigurationError(f"precision config candidates cannot be applied: {exc}") from exc
        elif calibration_data is not None:
            if not isinstance(calibration_data, CalibrationData):
                raise ConfigurationError("calibration_data must be a CalibrationData object prepared outside accelerate()")
            if dataset_name is not None and dataset_name != calibration_data.dataset_name:
                raise ConfigurationError("dataset_name does not match calibration_data")
            if acceleration_config.budget <= 0:
                raise ConfigurationError("budget must be a positive CAIEC epsilon when selecting precision")
            device_capability = torch.cuda.get_device_capability(target)
            selection = select_balle_precision(
                runtime_model, catalog, calibration_data,
                epsilon=acceleration_config.budget, p_ch=p_ch,
                model_name=type(model).__qualname__, device_capability=device_capability,
                coder=selected_coder,
            )
            precision_selection_result = selection
            runtime_model = selection.model
            default_root = (
                Path(precision_config_output_dir)
                if precision_config_output_dir is not None
                else Path.home() / ".config" / "fast_compressai" / "precision"
            )
            config_path = PrecisionConfig.default_path(
                default_root, type(model).__qualname__, calibration_data.dataset_name,
            )
            selection = replace(
                selection,
                config=replace(
                    selection.config,
                    records_path=str(config_path.with_name(config_path.stem + "-records.json")),
                    recovery_path=str(config_path.with_name(config_path.stem + "-recovery.json")),
                ),
            )
            selection.config.save(config_path)
            records_path = config_path.with_name(config_path.stem + "-records.json")
            recovery_path = config_path.with_name(config_path.stem + "-recovery.json")
            _save_json(records_path, {
                "kind": "fast_compressai_precision_records",
                "selection_config": str(config_path),
                "records": list(selection.config.probe_records),
            })
            _save_json(recovery_path, {
                "kind": "fast_compressai_precision_recovery",
                "selection_config": str(config_path),
                "epsilon": selection.config.epsilon,
                "recovery_order": list(selection.config.recovery_order),
                "steps": list(selection.config.recovery_steps),
                "validated_rd_cost": selection.config.validated_rd_cost,
                "budget_met": selection.config.budget_met,
            })
            config_path_used = str(config_path)
            final_rd_cost = selection.rd_cost
            final_budget_met = selection.config.budget_met
            profile_probe_count = len(selection.config.probe_records)
            unavailable_probe_count = sum(
                1 for record in selection.config.probe_records
                if record.get("status") == "unavailable"
            )
            recovered_count = len(selection.config.recovery_order)
        p_ch = acceleration_config.p_ch if acceleration_config.p_ch is not None else 1
        inventory = catalog.inventory(runtime_model)
        transform_instances = tuple(
            instance for instance in inventory
            if instance.logical_op in {"conv2d", "conv_transpose2d", "gdn", "igdn"}
        )
        codec = FactorizedWarpDivCodec(runtime_model, p_ch)
        compile_config = CompileConfig(device=str(target))
    else:
        runtime_model = model
    selected_candidate = "fp32"
    profiling_status = "not_run"
    selection_status = "not_run"
    psnr_drop_db = None
    reference_latency = None
    candidate_latency = None
    runtime_status = "not_applied"
    speedup = None
    message = (
        "Reference wrapper returned. P was recorded as GA-ANS p_ch; no precision candidate was applied."
    )
    if sample is not None and selected_coder == "rans":
        if not isinstance(sample, torch.Tensor):
            raise TypeError("sample must be a torch.Tensor when provided")
        try:
            candidate_model, replaced = make_bf16_conv_candidate(model)
            if replaced == 0:
                profiling_status = "unavailable"
                selection_status = "fallback_fp32"
                message = "No supported Conv2d candidate was found; retained FP32 reference."
            else:
                profile = profile_forward_pair(model, sample, candidate=candidate_model)
                profiling_status = "measured"
                psnr_drop_db = profile.candidate.psnr_drop_db
                reference_latency = profile.reference.latency_ms
                candidate_latency = profile.candidate.latency_ms
                quality_ok = psnr_drop_db is not None and psnr_drop_db <= acceleration_config.budget
                faster = (
                    reference_latency is not None
                    and candidate_latency is not None
                    and candidate_latency < reference_latency
                )
                if quality_ok and faster:
                    selected_candidate = "bf16"
                    selection_status = "selected"
                    runtime_model = candidate_model
                    runtime_status = "applied"
                    speedup = reference_latency / candidate_latency
                    message = (
                        "BF16 Conv2d candidate passed the forward PSNR-drop budget and was measured faster; "
                        "the candidate model is applied to the returned wrapper. CompressAI RANS remains "
                        "the entropy coder and GA-ANS p_ch is not applied."
                    )
                else:
                    selection_status = "fallback_fp32"
                    message = (
                        "BF16 candidate was measured but retained FP32 because the quality budget "
                        "or measured latency condition was not satisfied."
                    )
        except (RuntimeError, ValueError) as exc:
            profiling_status = "failed"
            selection_status = "fallback_fp32"
            message = f"Candidate profiling unavailable ({exc}); retained FP32 reference."
    wrapped = compile(runtime_model, config=compile_config)
    if codec is not None:
        wrapped._coder = codec
        wrapped.operator_inventory = inventory
        wrapped.operator_candidates = transform_instances
        message = (
            f"FactorizedPrior inventory prepared {len(transform_instances)} GPU floating operator instances. "
            "The local warp_div GA-ANS backend is applied to compress/decompress and P_ch is applied. "
            "Payloads remain device-local; entropy coder RD selection is reported separately."
        )
        if precision_selection_result is not None:
            message = (
                f"CAIEC operator selection applied {len(precision_selection_result.config.assignments)} "
                "per-instance assignments; full RD cost was measured and the precision config was saved. "
                + message
            )
    if precision_config_path is not None:
        selected_candidate = "config"
        selection_status = "config_reused"
        runtime_status = "applied"
        profiling_status = "loaded"
        message = "Precision config loaded and applied without re-profiling or re-selection."
    elif precision_selection_result is not None:
        selected_candidate = "caiec-mixed"
        selection_status = "selected" if final_budget_met else "budget_infeasible"
        runtime_status = "applied"
        profiling_status = "measured"
        message = (
            f"CAIEC selected and applied {len(precision_selection_result.applications)} operator-instance candidates; "
            f"full RD Cost={final_rd_cost:.6g}, budget_met={final_budget_met}."
        )

    report = AccelerationReport(
        status="coder_applied" if codec is not None else ("reference_only" if runtime_status != "applied" else "accelerated"),
        model_class=type(model).__qualname__,
        adapter_family=wrapped.report.contract_family,
        p_ch=codec.p_ch if codec is not None else acceleration_config.p_ch,
        quality_budget=float(acceleration_config.budget),
        quality_metric="psnr_drop_db",
        profiling_status=profiling_status,
        precision_selection_status=("config_applied" if precision_config_path is not None else ("selected" if precision_selection_result is not None else selection_status)),
        selected_candidate=selected_candidate,
        psnr_drop_db=psnr_drop_db,
        reference_latency_ms=reference_latency,
        candidate_latency_ms=candidate_latency,
        coder_status="warp_div" if codec is not None else "rans",
        p_ch_applied=codec is not None,
        runtime_status=runtime_status,
        speedup=speedup,
        message=message,
        rd_cost=final_rd_cost,
        budget_met=final_budget_met,
        precision_config_path=config_path_used,
        sensitivity_probe_count=profile_probe_count,
        unavailable_probe_count=unavailable_probe_count,
        recovered_instance_count=recovered_count,
    )
    wrapped.acceleration_config = acceleration_config
    wrapped.acceleration_report = report
    return wrapped


class FastModel(nn.Module):
    """Reference wrapper that preserves a model's public compression paths."""

    def __init__(
        self,
        model: nn.Module,
        config: CompileConfig,
        *,
        report: CompileReport | None = None,
        registry: OperatorRegistry | None = None,
        adapter: CompressAIAdapter | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.config = config
        self._registry = registry
        self._adapter = adapter or CompressAIAdapter(model)
        available = tuple(
            name for name in ("forward", "compress", "decompress") if hasattr(model, name)
        )
        missing = tuple(name for name in ("compress", "decompress") if not hasattr(model, name))
        self._report = report or CompileReport(
            backend=config.backend,
            accelerated=False,
            available_paths=available,
            missing_paths=missing,
            fallback_used=bool(missing) and config.enable_reference_fallback,
            message=(
                "Reference backend selected; no low-precision or graph optimization "
                "was applied."
            ),
        )
        if config.device is not None:
            self.to(config.device)

    @property
    def adapter(self) -> CompressAIAdapter:
        return self._adapter

    @property
    def report(self) -> CompileReport:
        return self._report

    @property
    def operator_registry(self) -> OperatorRegistry | None:
        return self._registry

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._adapter.forward(*args, **kwargs)

    def compress(self, *args: Any, **kwargs: Any) -> Any:
        coder = self.__dict__.get("_coder")
        if coder is not None:
            return coder.compress(*args, **kwargs)
        return self._adapter.compress(*args, **kwargs)

    def decompress(self, *args: Any, **kwargs: Any) -> Any:
        coder = self.__dict__.get("_coder")
        if coder is not None:
            return coder.decompress(*args, **kwargs)
        return self._adapter.decompress(*args, **kwargs)

    def update(self, *args: Any, **kwargs: Any) -> Any:
        result = self._adapter.update(*args, **kwargs)
        self._report = replace(self._report, update_called=True)
        return result

    def state_dict(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.load_state_dict(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            modules = object.__getattribute__(self, "_modules")
            model = modules.get("model")
            if model is not None:
                return getattr(model, name)
            raise


def _normalize_operator_requests(
    operators: Sequence[str | Mapping[str, Any]] | None,
) -> list[tuple[str, str | None, str | None, str | None, OperatorContext | Mapping[str, Any] | None]]:
    if operators is None:
        return []
    if isinstance(operators, (str, bytes)) or not isinstance(operators, Sequence):
        raise TypeError("operators must be a sequence of string keys or mappings")

    requests = []
    for request in operators:
        if isinstance(request, str):
            requests.append((request, None, None, None, None))
            continue
        if not isinstance(request, Mapping):
            raise TypeError("each operator request must be a string key or mapping")
        unknown = set(request) - {"key", "precision", "context", "component"}
        if unknown:
            names = ", ".join(sorted(map(str, unknown)))
            raise TypeError(f"unknown operator request field(s): {names}")
        key = request.get("key")
        if not isinstance(key, str) or not key.strip():
            raise TypeError("operator request key must be a non-empty string")
        requests.append((key, request.get("precision"), request.get("component"), "explicit_operator" if request.get("precision") is not None else None, request.get("context")))
    return requests


def _build_report(
    model: nn.Module,
    config: CompileConfig,
    decisions: tuple[OperatorDecision, ...],
    adapter: CompressAIAdapter,
    policy: PrecisionPolicy,
    precision_sources: Mapping[str, str],
) -> CompileReport:
    available = tuple(name for name in ("forward", "compress", "decompress") if hasattr(model, name))
    missing = tuple(name for name in ("compress", "decompress") if not hasattr(model, name))
    operator_fallback = any(decision.fallback_used for decision in decisions)
    accelerated = any(decision.accelerated for decision in decisions)
    precision_map = {decision.key: decision.selected_precision for decision in decisions}
    requested = {decision.key: decision.requested_precision or "fp32" for decision in decisions}
    sources = {key: precision_sources.get(key, "default") for key in precision_map}
    if not decisions:
        message = "Reference backend selected; precision policy evaluated for reporting only."
    elif operator_fallback:
        message = "Precision policy evaluation completed with reference fallbacks."
    elif accelerated:
        message = "Precision policy evaluation selected non-reference implementations."
    else:
        message = "Precision policy evaluation selected reference implementations."
    info = adapter.info
    message = f"{message} Adapter contract={info.contract_family}; no runtime precision conversion or graph rewriting was performed."
    serialized = policy.to_dict() if policy.serializable else None
    return CompileReport(
        backend=config.backend,
        accelerated=False,
        available_paths=available,
        missing_paths=missing,
        fallback_used=(bool(missing) and config.enable_reference_fallback) or operator_fallback,
        message=message,
        operator_decisions=decisions,
        operator_fallback_used=operator_fallback,
        candidate_accelerated=accelerated,
        runtime_applied=False,
        framework=info.framework,
        framework_version=info.framework_version,
        model_class=info.model_class,
        model_module=info.model_module,
        contract_family=info.contract_family,
        update_available=info.update_available,
        update_called=info.update_called,
        entropy_models=info.entropy_models,
        signature_level=info.signature_level,
        compress_signature=info.compress_signature,
        decompress_signature=info.decompress_signature,
        validation_level=info.validation_level,
        precision_policy_type=policy.kind.value,
        precision_policy_serialized=serialized,
        precision_map=precision_map,
        precision_sources=sources,
        requested_precisions=requested,
        selected_precisions=precision_map,
        precision_fallback_used=operator_fallback,
    )


def compile(
    model: nn.Module,
    config: CompileConfig | Mapping[str, Any] | None = None,
    operators: Sequence[str | Mapping[str, Any]] | None = None,
    registry: OperatorRegistry | None = None,
    precision_policy: PrecisionPolicy | None = None,
) -> FastModel:
    """Compile through a reference path and optionally evaluate a precision policy."""

    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an instance of torch.nn.Module, got {type(model).__name__}")
    normalized_config = CompileConfig.from_value(config)
    if precision_policy is None:
        precision_policy = PrecisionPolicy.fp32()
    if not isinstance(precision_policy, PrecisionPolicy):
        raise TypeError("precision_policy must be a PrecisionPolicy instance or None")
    active_registry = registry or default_registry
    adapter = CompressAIAdapter(model)
    requests = _normalize_operator_requests(operators)
    decisions: list[OperatorDecision] = []
    precision_sources: dict[str, str] = {}
    for key, explicit_precision, component, explicit_source, context in requests:
        if context is None and normalized_config.device is not None:
            context = OperatorContext(device=normalized_config.device)
        resolution = precision_policy.resolve(
            PrecisionRequest(component=component, operator=key, context=context)
        )
        precision = explicit_precision or resolution.precision
        precision_sources[key] = explicit_source or resolution.source
        decision, _ = active_registry.select(
            key,
            precision=precision,
            context=context,
            allow_fallback=normalized_config.enable_operator_fallback,
        )
        decisions.append(decision)
    report = _build_report(model, normalized_config, tuple(decisions), adapter, precision_policy, precision_sources)
    return FastModel(model, normalized_config, report=report, registry=active_registry, adapter=adapter)


def optimize(
    model: nn.Module,
    config: CompileConfig | Mapping[str, Any] | None = None,
    operators: Sequence[str | Mapping[str, Any]] | None = None,
    registry: OperatorRegistry | None = None,
    precision_policy: PrecisionPolicy | None = None,
) -> FastModel:
    """Compatibility alias for ``compile`` during the reference-only MVP."""

    return compile(model, config=config, operators=operators, registry=registry, precision_policy=precision_policy)
