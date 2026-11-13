"""Bounded visual dust emitted only by loaded feet disturbing nearby grains.

Dust is a visual, one-way approximation. Rigid grains provide actual contact.
"""

import numpy as np

from mars_project.environment.regolith import terrain_height


def disturbed_grains(positions, velocities, feet, forces):
    """Select moving grains close to an actually loaded foot, never ambient wind."""
    loaded = np.linalg.norm(forces, axis=-1) > 2.0
    if not loaded.any():
        return np.zeros(len(positions), dtype=bool)
    distance = np.linalg.norm(positions[:, None, :] - feet[None, loaded, :], axis=-1)
    return (distance.min(axis=1) < 0.22) & (np.linalg.norm(velocities, axis=-1) > 0.04)


def drag_velocity(velocity, dt, air_density, radius=50e-6, particle_density=1500.0, drag_coefficient=1.0):
    """Stable quadratic drag for a spherical dust proxy in stationary air.

    This is not a rarefied-gas or resolved aerosol model. Visual sizes are enlarged.
    """
    k = 3 * drag_coefficient * air_density / (8 * radius * particle_density)
    return velocity / (1 + k * np.linalg.norm(velocity, axis=-1, keepdims=True) * dt)


class DustCloud:
    """Fixed-size particle pool; bounded lifetime, emissions and spatial extent."""

    def __init__(self, profile, gravity=-3.71, capacity=512, seed=42):
        self.profile = profile
        self.gravity = gravity
        self.rng = np.random.default_rng(seed)
        self.positions = np.zeros((capacity, 3))
        self.velocities = np.zeros((capacity, 3))
        self.life = np.zeros(capacity)
        self.total_emitted = 0
        self.emission_credit = 0.0

    def clear(self):
        self.life[:] = 0
        self.emission_credit = 0.0

    def update(self, dt, grain_positions, grain_velocities, feet, forces):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Dust timestep must be positive and finite")
        alive = self.life > 0
        self.life[alive] -= dt
        # Small integration substeps keep gravity/ground handling stable.
        substeps = max(1, int(np.ceil(dt / 0.005)))
        h = dt / substeps
        for _ in range(substeps):
            self.velocities[alive, 2] += self.gravity * h
            self.velocities[alive] = drag_velocity(self.velocities[alive], h, self.profile.air_density)
            self.positions[alive] += self.velocities[alive] * h
        p = self.positions
        outside = (p[:, 0] < -1.5) | (p[:, 0] > 5) | (np.abs(p[:, 1]) > 2)
        landed = p[:, 2] <= terrain_height(p[:, 0], p[:, 1]) + 0.002
        self.life[outside | landed] = 0
        disturbed = disturbed_grains(grain_positions, grain_velocities, feet, forces)
        candidates = np.flatnonzero(disturbed)
        if not len(candidates):
            self.emission_credit = 0.0
            return
        # At most 160 emissions per simulation second, irrespective of frame rate.
        speed = np.linalg.norm(grain_velocities[candidates], axis=-1).max()
        self.emission_credit += min(160.0, 40 + 160 * speed) * dt
        count = int(self.emission_credit)
        self.emission_credit -= count
        slots = np.flatnonzero(self.life <= 0)[:count]
        selected = self.rng.choice(candidates, len(slots), replace=True)
        self.positions[slots] = grain_positions[selected] + self.rng.uniform(-0.01, 0.01, (len(slots), 3))
        self.positions[slots, 2] += 0.03
        self.velocities[slots] = grain_velocities[selected] + self.rng.normal(0, 0.08, (len(slots), 3))
        self.velocities[slots, 2] = np.abs(self.velocities[slots, 2]) + self.rng.uniform(0.2, 0.5, len(slots))
        self.life[slots] = self.rng.uniform(0.5, 1.2, len(slots))
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
            f"DUST_CONFIG profile={profile.name} air_density={profile.air_density} capacity=512 foot_contact_only=True",
            flush=True,
        )

    def clear(self):
        self.cloud.clear()
        self.render()

    def update(self, dt):
        grains = self.env.scene["regolith"].data
        self.cloud.update(
            dt,
            grains.body_link_pos_w.torch[0].detach().cpu().numpy(),
            grains.body_com_lin_vel_w.torch[0].detach().cpu().numpy(),
            self.robot.data.body_link_pos_w.torch[0, self.foot_ids].detach().cpu().numpy(),
            self.contact.data.net_forces_w.torch[0, self.contact_ids].detach().cpu().numpy(),
        )
        self.render()

    def render(self):
        from pxr import Sdf, Vt

        # Fixed-size arrays avoid transient mismatched/empty Fabric buffers.
        positions = self.cloud.positions.astype(np.float32)
        sizes = np.repeat((0.003 * np.clip(self.cloud.life / 0.2, 0.1, 1))[:, None], 3, axis=1).astype(np.float32)
        with Sdf.ChangeBlock():
            self.instancer.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(positions))
            self.instancer.GetScalesAttr().Set(Vt.Vec3fArray.FromNumpy(sizes))
            self.instancer.GetInvisibleIdsAttr().Set(np.flatnonzero(self.cloud.life <= 0).tolist())
