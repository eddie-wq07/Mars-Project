"""Apply aerodynamic loads through Isaac Lab's additive external-wrench path."""

import json
import math

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from mars_project.environment.aerodynamics import air_density, box_drag_force, link_box_dimensions


class AirDrag(ActionTerm):
    """Zero-input environment force term; preserves the pretrained action space.

    Forces refresh at each action-application tick (normally 5 ms) and are held
    through solver substeps. If a backend owns four-step decimation, refresh is
    20 ms instead. World forces at link CoMs add to other external loads.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.density = air_density(cfg.profile)
        if not math.isfinite(cfg.coefficient) or cfg.coefficient < 0:
            raise ValueError("Drag coefficient must be finite and nonnegative")
        if len(cfg.wind) != 3 or not all(math.isfinite(v) for v in cfg.wind):
            raise ValueError("Wind must be a finite world-frame 3-vector")
        self.dimensions = link_box_dimensions(env.sim.stage, self._asset.cfg.prim_path, self._asset.body_names).to(
            env.device
        )
        self.wind = torch.tensor(cfg.wind, device=env.device)
        self._empty = torch.empty((env.num_envs, 0), device=env.device)
        self.force_w = torch.zeros((env.num_envs, self._asset.num_bodies, 3), device=env.device)
        env.air_drag = self
        print(
            "AIR_DRAG_CONFIG "
            + json.dumps(
                dict(
                    profile=cfg.profile,
                    density_kg_m3=self.density,
                    coefficient=cfg.coefficient,
                    wind_m_s=cfg.wind,
                    force_refresh_s=(env.step_dt if env.sim.physics_manager.handles_decimation() else env.physics_dt),
                    area_model="per-link visual bounding box projected onto relative airflow",
                    body_dimensions_m=dict(zip(self._asset.body_names, self.dimensions.cpu().tolist(), strict=True)),
                    force_location="link center of mass",
                    actuator_damping_modified=False,
                ),
                sort_keys=True,
            ),
            flush=True,
        )

    @property
    def action_dim(self):
        return 0

    @property
    def raw_actions(self):
        return self._empty

    @property
    def processed_actions(self):
        return self._empty

    def process_actions(self, actions):
        pass

    def apply_actions(self):
        self.force_w.copy_(
            box_drag_force(
                self._asset.data.body_com_vel_w.torch[..., :3],
                self._asset.data.body_link_pose_w.torch[..., 3:],
                self.dimensions,
                self.density,
                self.cfg.coefficient,
                self.wind,
            )
        )
        self._asset.instantaneous_wrench_composer.add_forces_and_torques_index(forces=self.force_w, is_global=True)

    def reset(self, env_ids=None):
        # Articulation reset clears the wrench buffers; only clear our telemetry.
        self.force_w[slice(None) if env_ids is None else env_ids] = 0


@configclass
class AirDragCfg(ActionTermCfg):
    class_type: type = AirDrag
    asset_name: str = "robot"
    profile: str = "mars"
    coefficient: float = 1.0
    wind: tuple[float, float, float] = (0.0, 0.0, 0.0)


def configure_air_drag(cfg, profile="auto"):
    """Select robot atmosphere independently of backdrop and motor gains."""
    if profile == "auto":
        profile = "mars" if abs(cfg.sim.gravity[2]) < 5.0 else "earth"
    air_density(profile)  # Fail early for invalid configuration.
    cfg.actions.air_drag = AirDragCfg(profile=profile)
