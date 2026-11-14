"""Bounded visual dust emitted only by loaded feet disturbing nearby grains.

Dust is a visual, one-way approximation. A pressure-driven sand surface provides actual contact.
"""

import numpy as np

from mars_project.environment.regolith import X_MAX, X_MIN, Y_MAX, Y_MIN, terrain_height


def drag_velocity(velocity, dt, air_density, radius=50e-6, particle_density=1500.0, drag_coefficient=1.0):
    """Stable quadratic drag for a spherical dust proxy in stationary air.

    This is not a rarefied-gas or resolved aerosol model. Visible sizes match these particle radii.
    """
    k = 3 * drag_coefficient * air_density / (8 * radius * particle_density)
    return velocity / (1 + k * np.linalg.norm(velocity, axis=-1, keepdims=True) * dt)


class DustCloud:
    """Fine 0.15–0.35 mm specks, with a bounded pool and no ambient emission."""

    def __init__(self, profile, gravity=-3.71, capacity=2048, seed=42):
        self.profile = profile
        self.gravity = gravity
        self.rng = np.random.default_rng(seed)
        self.positions = np.zeros((capacity, 3))
        self.velocities = np.zeros((capacity, 3))
        self.radii = self.rng.uniform(0.000075, 0.000175, capacity)
        self.life = np.zeros(capacity)
        self.total_emitted = 0
        self.emission_credit = 0.0
        self.was_loaded = None

    def clear(self):
        self.life[:] = 0
        self.emission_credit = 0.0
        self.was_loaded = None

    def update(self, dt, feet, velocities, forces, surface_height=terrain_height, yaws=None):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Dust timestep must be positive and finite")
        alive = self.life > 0
        self.life[alive] -= dt
        n = max(1, int(np.ceil(dt / 0.005)))
        h = dt / n
        for _ in range(n):
            self.velocities[alive, 2] += self.gravity * h
            self.velocities[alive] = drag_velocity(
                self.velocities[alive], h, self.profile.air_density, radius=self.radii[alive, None]
            )
            self.positions[alive] += self.velocities[alive] * h
        p = self.positions
        outside = (p[:, 0] < -1.5) | (p[:, 0] > 5) | (np.abs(p[:, 1]) > 2)
        landed = p[:, 2] <= surface_height(p[:, 0], p[:, 1]) + self.radii
        self.life[outside | landed] = 0
        in_sand = (feet[:, 0] >= X_MIN) & (feet[:, 0] < X_MAX) & (feet[:, 1] >= Y_MIN) & (feet[:, 1] < Y_MAX)
        loaded = (forces[:, 2] > 5) & in_sand
        previous = np.zeros(len(feet), dtype=bool) if self.was_loaded is None else self.was_loaded
        active = loaded & ((np.linalg.norm(velocities, axis=-1) > 0.025) | ~previous)
        self.was_loaded = loaded.copy()
        candidates = np.flatnonzero(active)
        if not len(candidates):
            self.emission_credit = 0.0
            return
        self.emission_credit += 600.0 * dt
        count = int(self.emission_credit)
        self.emission_credit -= count
        slots = np.flatnonzero(self.life <= 0)[:count]
        source = self.rng.choice(candidates, len(slots), replace=True)
        uv = self.rng.uniform([-0.10, -0.045], [0.13, 0.045], (len(slots), 2))
        yaw = np.zeros(len(feet)) if yaws is None else yaws
        c, s = np.cos(yaw[source]), np.sin(yaw[source])
        self.positions[slots, :2] = feet[source, :2] + np.stack(
            (c * uv[:, 0] - s * uv[:, 1], s * uv[:, 0] + c * uv[:, 1]), axis=-1
        )
        self.positions[slots, 2] = surface_height(self.positions[slots, 0], self.positions[slots, 1]) + 0.001
        self.velocities[slots] = velocities[source] * 0.3 + self.rng.normal(0, 0.04, (len(slots), 3))
        self.velocities[slots, 2] = self.rng.uniform(0.12, 0.35, len(slots))
        self.life[slots] = self.rng.uniform(0.4, 0.9, len(slots))
        self.total_emitted += len(slots)


class FootstepDust:
    """Read contact/grain state after physics, then update non-colliding USD visuals."""

    def __init__(self, env, profile):
        from pxr import Sdf, UsdGeom, UsdShade

        from isaaclab.sim.utils.stage import get_current_stage

        self.env = env
        self.robot = env.scene["robot"]
        self.contact = env.scene["contact_forces"]
        self.foot_ids, _ = self.robot.find_bodies(".*_ankle_link")
        self.contact_ids, _ = self.contact.find_bodies(".*_ankle_link")
        if len(self.foot_ids) != 2 or len(self.contact_ids) != 2:
            raise RuntimeError("Footstep dust requires both H1 ankle contact bodies")
        self.cloud = DustCloud(profile, gravity=env.cfg.sim.gravity[2])
        stage = get_current_stage()
        self.instancer = UsdGeom.PointInstancer.Define(stage, "/World/FootstepDust")
        sphere = UsdGeom.Sphere.Define(stage, "/World/FootstepDust/Particle")
        sphere.CreateRadiusAttr().Set(1.0)
        material = UsdShade.Material.Define(stage, "/World/FootstepDust/Material")
        shader = UsdShade.Shader.Define(stage, "/World/FootstepDust/Material/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.36, 0.19, 0.08))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(material)
        self.instancer.GetPrototypesRel().SetTargets([sphere.GetPath()])
        self.instancer.GetProtoIndicesAttr().Set([0] * len(self.cloud.life))
        self.instancer.GetIdsAttr().Set(list(range(len(self.cloud.life))))
        self.render()
        print(
            f"DUST_CONFIG profile={profile.name} air_density={profile.air_density} "
            "capacity=2048 diameter_mm=0.15-0.35 foot_contact_only=True",
            flush=True,
        )

    def clear(self):
        self.cloud.clear()
        self.render()

    def update(self, dt):
        pose = self.robot.data.body_link_pose_w.torch[0, self.foot_ids].detach().cpu().numpy()
        q = pose[:, 3:7]
        yaw = np.arctan2(2 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]), 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2))
        self.cloud.update(
            dt,
            pose[:, :3],
            self.robot.data.body_link_lin_vel_w.torch[0, self.foot_ids].detach().cpu().numpy(),
            self.contact.data.net_forces_w.torch[0, self.contact_ids].detach().cpu().numpy(),
            surface_height=self.env.sand.field.sample,
            yaws=yaw,
        )
        self.render()

    def render(self):
        from pxr import Sdf, Vt

        # Fixed-size arrays avoid transient mismatched/empty Fabric buffers.
        positions = self.cloud.positions.astype(np.float32)
        sizes = np.repeat((self.cloud.radii * np.clip(self.cloud.life / 0.1, 0.1, 1))[:, None], 3, axis=1).astype(
            np.float32
        )
        with Sdf.ChangeBlock():
            self.instancer.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(positions))
            self.instancer.GetScalesAttr().Set(Vt.Vec3fArray.FromNumpy(sizes))
            self.instancer.GetInvisibleIdsAttr().Set(np.flatnonzero(self.cloud.life <= 0).tolist())
