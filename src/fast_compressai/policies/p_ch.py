"""GA-ANS channel chunk-size (P_ch) selection policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class PChSelection:
    selected_p_ch: int
    status: str
    reason: str
    relative_error: float
    plateau: Mapping[str, object]


def legal_p_candidates(channel_count: int, candidates: Sequence[int] | None = None) -> tuple[int, ...]:
    if type(channel_count) is not int or channel_count < 1:
        raise ValueError("channel_count must be a positive integer")
    values = range(1, channel_count + 1) if candidates is None else candidates
    result = sorted({value for value in values if type(value) is int and 1 <= value <= channel_count})
    if not result:
        raise ValueError("no legal P_ch candidates")
    return tuple(result)


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def choose_p_ch(
    measurements: Sequence[Mapping[str, object]],
    *,
    relative_tolerance: float = 0.10,
    plateau_delta: float = 0.05,
    plateau_patience: int = 2,
) -> PChSelection:
    """Choose the largest P whose codec/inference ratio is balanced.

    Measurements must contain ``p_ch``, ``inference_total_ms`` and
    ``codec_total_ms``. Candidates are considered from largest to smallest
    P so that the selected point favors the lower-bpp side of the trade-off.
    Plateau detection is diagnostic and does not treat noisy tail changes as
    meaningful algorithmic improvements.
    """
    if not 0 <= relative_tolerance < 1:
        raise ValueError("relative_tolerance must be in [0, 1)")
    if not 0 <= plateau_delta:
        raise ValueError("plateau_delta must be non-negative")
    if type(plateau_patience) is not int or plateau_patience < 1:
        raise ValueError("plateau_patience must be positive")
    usable = []
    for item in measurements:
        p_ch = item.get("p_ch")
        if type(p_ch) is not int or p_ch < 1:
            continue
        forward = _finite_positive(item.get("inference_total_ms"), "inference_total_ms")
        codec = _finite_positive(item.get("codec_total_ms"), "codec_total_ms")
        ratio = codec / forward
        error = abs(ratio - 1.0)
        usable.append((p_ch, ratio, error, codec))
    if not usable:
        raise ValueError("no usable P_ch measurements")
    usable.sort()
    plateau_pairs = []
    for previous, current in zip(usable, usable[1:]):
        delta = abs(current[3] - previous[3]) / max(previous[3], 1e-12)
        plateau_pairs.append({"from_p_ch": previous[0], "to_p_ch": current[0], "relative_delta": delta})
    plateau = {"detected": False, "pairs": plateau_pairs}
    run = 0
    for pair in plateau_pairs:
        run = run + 1 if pair["relative_delta"] <= plateau_delta else 0
        if run >= plateau_patience:
            plateau["detected"] = True
            plateau["start_p_ch"] = plateau_pairs[plateau_pairs.index(pair) - run + 1]["from_p_ch"]
            plateau["end_p_ch"] = pair["to_p_ch"]
            plateau["interpretation"] = "algorithmic improvement is not distinguishable from measurement variation"
            break
    passing = [item for item in usable if item[2] <= relative_tolerance]
    if passing:
        selected = max(passing, key=lambda item: item[0])
        return PChSelection(selected[0], "auto_selected", "largest_relative_error_pass", selected[2], plateau)
    selected = min(usable, key=lambda item: (item[2], -item[0]))
    return PChSelection(selected[0], "fallback_closest_balance", "no_candidate_within_relative_tolerance", selected[2], plateau)
