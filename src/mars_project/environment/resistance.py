"""Legacy motor-damping heuristic, disabled by default; not aerodynamics."""

import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EnvironmentProfile:
    name: str
    air_density: float  # kg/m^3; also used by the independent dust drag model
    added_joint_damping: float  # N m s/rad, at resistance scale 1


EARTH = EnvironmentProfile("earth", 1.225, 0.2)
MARS = EnvironmentProfile("mars", 0.020, 0.2 * 0.020 / 1.225)


def environment_profile(name):
    if name not in ("earth", "mars"):
        raise ValueError(f"Unknown environment profile: {name}")
    return EARTH if name == "earth" else MARS


def apply_joint_resistance(cfg, profile, scale=0.0):
    """Add a small damping increment without replacing tuned baseline gains.

    The density ratio is a heuristic, not a joint-level aerodynamic derivation.
    Scale 0 preserves the checkpoint's original motor configuration exactly.
    Call once on a fresh environment config, before constructing actuators.
    """
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("Joint resistance scale must be finite and nonnegative")
    if getattr(cfg, "_joint_resistance_applied", False):
        raise ValueError("Joint resistance already applied; start from a fresh config")
    addition = profile.added_joint_damping * scale
    baseline = {}
    for name, actuator in cfg.scene.robot.actuators.items():
        damping = actuator.damping
        if damping is None:
            raise ValueError(f"{name} has no explicit baseline damping")
        baseline[name] = dict(damping) if isinstance(damping, dict) else damping
    for name, actuator in cfg.scene.robot.actuators.items():
        damping = baseline[name]
        actuator.damping = (
            {key: value + addition for key, value in damping.items()}
            if isinstance(damping, dict)
            else damping + addition
        )
    cfg._joint_resistance_applied = True
    summary = dict(
        profile=profile.name,
        air_density_kg_m3=profile.air_density,
        scale=scale,
        added_joint_damping=addition,
        baseline=baseline,
        model="added actuator damping; not physical air drag",
    )
    print("RESISTANCE_CONFIG " + json.dumps(summary, sort_keys=True), flush=True)
    return summary
