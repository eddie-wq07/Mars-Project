"""Reproducible paired starts and explicit trajectory scoring, independent of Kit."""

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Protocol:
    goal_m: float = 5.0
    tolerance_m: float = 0.5
    timeout_s: float = 30.0
    command_m_s: float = 0.5
    position_jitter_m: float = 0.05
    yaw_jitter_rad: float = math.radians(3.0)
    velocity_jitter_m_s: float = 0.05
    yaw_rate_jitter_rad_s: float = 0.02
    minimum_root_height_m: float = 0.55
    minimum_up_cosine: float = 0.5

    def to_dict(self):
        return asdict(self)


DEFAULT_PROTOCOL = Protocol()


def paired_start(seed, protocol=DEFAULT_PROTOCOL):
    rng = np.random.default_rng(seed)
    return dict(
        seed=seed,
        forward_m=float(rng.uniform(-protocol.position_jitter_m, protocol.position_jitter_m)),
        lateral_m=float(rng.uniform(-protocol.position_jitter_m, protocol.position_jitter_m)),
        yaw_rad=float(rng.uniform(-protocol.yaw_jitter_rad, protocol.yaw_jitter_rad)),
        forward_velocity_m_s=float(rng.uniform(-protocol.velocity_jitter_m_s, protocol.velocity_jitter_m_s)),
        lateral_velocity_m_s=float(rng.uniform(-protocol.velocity_jitter_m_s, protocol.velocity_jitter_m_s)),
        yaw_rate_rad_s=float(rng.uniform(-protocol.yaw_rate_jitter_rad_s, protocol.yaw_rate_jitter_rad_s)),
    )


def route_rotation(yaw):
    return np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])


def classify_sample(position_xy, target_xy, root_height, up_cosine, seconds, inside, protocol=DEFAULT_PROTOCOL):
    """A fall/boundary exit overrides success; a goal reached at the deadline counts."""
    values = [*position_xy, *target_xy, root_height, up_cosine, seconds]
    if not np.isfinite(values).all():
        return "invalid_state"
    if root_height < protocol.minimum_root_height_m or up_cosine < protocol.minimum_up_cosine:
        return "fall"
    if not inside:
        return "out_of_bounds"
    if np.linalg.norm(np.asarray(position_xy) - target_xy) <= protocol.tolerance_m:
        return "success"
    if seconds >= protocol.timeout_s - 1e-9:
        return "timeout"
    return None


def summarize(results):
    counts = {
        name: sum(r["outcome"] == name for r in results)
        for name in ("success", "fall", "timeout", "out_of_bounds", "invalid_state")
    }
    arrivals = [r["duration_s"] for r in results if r["outcome"] == "success"]
    n = len(results)
    p = counts["success"] / n if n else 0.0
    z = 1.959963984540054
    center = (p + z * z / (2 * n)) / (1 + z * z / n) if n else 0
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n) if n else 0
    return dict(
        trials=n,
        outcomes=counts,
        success_rate=p,
        success_rate_wilson_95=[max(0, center - half), min(1, center + half)],
        successful_arrival_mean_s=float(np.mean(arrivals)) if arrivals else None,
        successful_arrival_median_s=float(np.median(arrivals)) if arrivals else None,
        successful_arrival_std_s=float(np.std(arrivals, ddof=1)) if len(arrivals) > 1 else None,
        mean_max_abs_lateral_drift_m=float(np.mean([r["max_abs_lateral_m"] for r in results])) if n else None,
    )
