"""Local active MPM regions, a continuous terrain skin, and conservative dust transfer.

Native Newton active flags exclude undisturbed material from grid assembly and
advection. A prescribed zero-velocity halo provides confining material at the
truncation boundary. This is a local-domain approximation, not global wave dynamics.
"""

import math

import numpy as np

from mars_project.environment.airborne_dust import Atmosphere, DustParcels
from mars_project.environment.mpm_soil import DENSITY, THICKNESS, VOXEL_SIZE, X_MAX, X_MIN, Y_MAX, Y_MIN
from mars_project.environment.regolith import terrain_height

# Fraction of a detached bulk sample released as suspended fines. This is an
# uncalibrated entrainment assumption, not the soil's total fine-grain content.
# Keep the resolved cloud dilute enough for prescribed-gas transport.
RELEASED_FINE_FRACTION = 1.0e-6


class SoilRuntime:
    def __init__(self, env):
        import torch
        import warp as wp

        from isaaclab_newton.physics import NewtonManager

        self.env, self.soil = env, env.scene["soil"]
        self.solver = NewtonManager._solver.solver("mpm")
        self.model = NewtonManager.get_model()
        self.mpm = self.solver._mpm_model
        self.torch, self.wp = torch, wp
        self.foot_ids, _ = env.scene["robot"].find_bodies(".*_ankle_link")
        self.initial = self.soil.data.default_particle_state_w.torch[0, :, :3].clone()
        self.nx, self.ny = math.ceil((X_MAX - X_MIN) / VOXEL_SIZE), math.ceil((Y_MAX - Y_MIN) / VOXEL_SIZE)
        self.base_mass = torch.full((len(self.initial),), self.soil.cfg.spawn.mass, device=env.device)
        self.base_volume = self.base_mass / DENSITY
        xs = X_MIN + (np.arange(self.nx) + 0.5) * VOXEL_SIZE
        ys = Y_MIN + (np.arange(self.ny) + 0.5) * VOXEL_SIZE
        xx, yy = np.meshgrid(xs, ys)
        self.bed = torch.as_tensor(terrain_height(xx, yy).astype(np.float32) - THICKNESS, device=env.device)
        self.reference = self.top_grid(self.initial)
        self.mass = self.base_mass.clone()
        self.quiet = torch.zeros(len(self.initial), device=env.device)
        self.displacement = torch.zeros_like(self.bed)
        self.deposited_height = torch.zeros_like(self.bed)
        gravity = env.cfg.sim.gravity[2]
        atmosphere = (
            Atmosphere(gravity=gravity)
            if abs(gravity) < 5
            else Atmosphere(
                pressure=101325.0, temperature=288.15, viscosity=1.81e-5, molar_mass=0.02897, gravity=gravity
            )
        )
        self.dust = DustParcels(atmosphere)
        self.state = NewtonManager._solver.entry_state("mpm")
        self.history = {
            name: wp.to_torch(getattr(self.state.mpm, name)).clone()
            for name in ("particle_elastic_strain", "particle_Jp", "particle_stress")
        }
        self.previous_dynamic = torch.zeros(len(self.initial), dtype=torch.bool, device=env.device)
        self.surface = None
        self.volume_renderer = None
        if env.sim.is_rendering:
            from mars_project.environment.soil_surface import SoilSurface

            self.surface = SoilSurface(env)
            from mars_project.environment.dust_volume import DustVolume

            self.volume_renderer = DustVolume(env)
        self.reset()

    def cell_ids(self, points):
        ix = ((points[:, 0] - X_MIN) / VOXEL_SIZE).floor().long()
        iy = ((points[:, 1] - Y_MIN) / VOXEL_SIZE).floor().long()
        inside = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return iy.clamp(0, self.ny - 1) * self.nx + ix.clamp(0, self.nx - 1), inside

    def top_grid(self, points):
        t = self.torch
        ids, valid = self.cell_ids(points)
        top = t.full((self.nx * self.ny,), -t.inf, device=points.device)
        top.scatter_reduce_(0, ids[valid], points[valid, 2] + self.soil.cfg.spawn.radius, reduce="amax")
        return t.maximum(top.reshape(self.ny, self.nx), self.bed)

    def displacement_at(self, xy):
        import torch.nn.functional as f

        coords = xy.clone()
        coords[:, 0] = 2 * (coords[:, 0] - X_MIN) / (self.nx * VOXEL_SIZE) - 1
        coords[:, 1] = 2 * (coords[:, 1] - Y_MIN) / (self.ny * VOXEL_SIZE) - 1
        return f.grid_sample(
            self.displacement[None, None], coords[None, None], align_corners=False, padding_mode="zeros"
        )[0, 0, 0]

    def heights_numpy(self, xy):
        coords = self.torch.as_tensor(xy, dtype=self.initial.dtype, device=self.initial.device)
        return terrain_height(xy[:, 0], xy[:, 1]) + self.displacement_at(coords).cpu().numpy()

    def reset(self):
        self.mass.copy_(self.base_mass)
        self.quiet.zero_()
        self.displacement.zero_()
        self.deposited_height.zero_()
        self.dust.reset()
        self.dust_loading = self.peak_dust_loading = 0.0
        self.previous_dynamic.zero_()
        for name, saved in self.history.items():
            if name == "particle_elastic_strain":
                # Newton stores the elastic deformation gradient here: the
                # undeformed state is I, not a zero small-strain tensor.
                saved.copy_(self.torch.eye(3, device=saved.device))
            else:
                saved.fill_(1.0 if name == "particle_Jp" else 0.0)
        self.apply_activity(0.0)
        if self.surface:
            self.surface.update(self)
            self.volume_renderer.update(self.dust)

    def apply_activity(self, dt):
        import torch.nn.functional as f

        t, wp = self.torch, self.wp
        points, velocities = self.soil.data.particle_pos_w.torch[0], self.soil.data.particle_vel_w.torch[0]
        feet = self.env.scene["robot"].data.body_link_pose_w.torch[0, self.foot_ids, :3]
        distance = ((points[:, None, :2] - feet[None, :, :2]) ** 2).sum(-1).min(1).values.sqrt()
        speed = velocities.norm(dim=-1)
        self.quiet = t.where(speed > 0.01, 0.0, self.quiet + dt)
        # Retain moving grains and their neighbouring material, including after a foot leaves.
        ids, valid = self.cell_ids(points)
        moving = t.zeros(self.nx * self.ny, device=points.device)
        moving.scatter_reduce_(
            0, ids[valid], ((speed[valid] > 0.01) | (self.quiet[valid] < 0.4)).float(), reduce="amax"
        )
        if dt == 0:
            moving.zero_()
            self.quiet.fill_(0.4)
        moving = f.max_pool2d(moving.reshape(1, 1, self.ny, self.nx), 9, 1, 4)[0, 0].flatten()
        dynamic = (distance < 0.55) | ((moving[ids] > 0) & valid)
        cells = t.zeros(self.nx * self.ny, device=points.device)
        cells.scatter_reduce_(0, ids[valid], dynamic[valid].float(), reduce="amax")
        halo = f.max_pool2d(cells.reshape(1, 1, self.ny, self.nx), 7, 1, 3)[0, 0].flatten()[ids] > 0
        active = dynamic | (halo & valid)
        # Newton clears history for particles outside its active quadrature.
        # Persist sleeping strain/plastic history, then restore it on reactivation.
        for name, saved in self.history.items():
            state = wp.to_torch(getattr(self.state.mpm, name))
            saved[self.previous_dynamic] = state[self.previous_dynamic]
            waking = active & ~self.previous_dynamic
            state[waking] = saved[waking]
        self.previous_dynamic.copy_(dynamic)
        self.dynamic, self.active = dynamic, active
        flags = active.to(t.int32)
        # Keep global/local masks consistent. Their arrays need not alias in Coupler.
        for model in (self.model, self.solver.model):
            wp.to_torch(model.particle_flags).copy_(flags)
            wp.to_torch(model.particle_mass).copy_(t.where(active & ~dynamic, 0.0, self.mass))
            wp.to_torch(model.particle_inv_mass).copy_(t.where(active & ~dynamic, 0.0, 1 / self.mass))
        wp.to_torch(self.mpm.particle_flags).copy_(flags)
        wp.to_torch(self.mpm.material_particle_flags).copy_(flags)
        # Newton defines density=0 as a prescribed-velocity particle boundary.
        # Physical mass remains in the ledger; only the solver halo has infinite inertia.
        wp.to_torch(self.mpm.particle_density).copy_(t.where(dynamic, DENSITY, 0.0))
        volume = self.mass / DENSITY
        wp.to_torch(self.mpm.particle_volume).copy_(volume)
        wp.to_torch(self.mpm.material_particle_volume).copy_(t.where(active, volume, 0.0))
        wp.to_torch(self.mpm.particle_radius).copy_((volume / 8) ** (1 / 3))
        velocities[~dynamic] = 0
        wp.to_torch(self.state.particle_qd)[~dynamic] = 0
        from isaaclab_newton.physics import NewtonManager

        wp.to_torch(NewtonManager.get_state_0().particle_qd)[~dynamic] = 0
        wp.to_torch(self.state.mpm.particle_qd_grad)[~dynamic] = 0

    def update(self, dt):
        import torch.nn.functional as f

        t = self.torch
        points, velocities = self.soil.data.particle_pos_w.torch[0], self.soil.data.particle_vel_w.torch[0]
        finite = points.isfinite().all(-1) & velocities.isfinite().all(-1)
        if not finite.all():
            bad = t.nonzero(~finite).flatten()
            raise RuntimeError(
                f"Local MPM became nonfinite: count={len(bad)} "
                f"dynamic={int(self.dynamic[bad].sum())} "
                f"initial_positions={self.initial[bad[:4]].tolist()}"
            )
        # A surface isolated above the local column average can supply a fine
        # fraction. Averaging neighbouring columns reduces self-screening.
        top = self.top_grid(points)
        neighbours = f.avg_pool2d(top[None, None], 5, 1, 2, count_include_pad=False)[0, 0]
        ids, valid = self.cell_ids(points)
        detached = valid & (points[:, 2] > neighbours.flatten()[ids] + 0.03) & (velocities[:, 2] > 0.05)
        available = (self.mass - self.base_mass * (1 - RELEASED_FINE_FRACTION)).clamp(min=0)
        candidates = t.nonzero(detached & (available > 1e-9)).flatten()[:64]
        if len(candidates):
            # Transfer at most the assumed released fine fraction, carrying the exact
            # source velocity: bulk+dust momentum and kinetic energy are unchanged.
            sources = candidates.cpu().numpy()
            accepted = self.dust.emit(
                points[candidates].cpu().numpy(),
                velocities[candidates].cpu().numpy(),
                available[candidates].cpu().numpy(),
                sources,
            )
            self.mass[accepted] -= available[accepted]
        top = self.top_grid(points[~detached])
        self.displacement = top - self.reference
        # Compact reconstruction kernel removes the voxel-sized stair steps. No
        # footprint prescription: only solved MPM displacements drive the surface.
        self.displacement = f.avg_pool2d(self.displacement[None, None], 3, 1, 1)[0, 0]
        self.displacement += self.deposited_height
        # Undisturbed terrain outside the local physics domain is the identical
        # source mesh. A buried three-cell guard remains at that source elevation.
        self.displacement[:3] = self.displacement[-3:] = 0
        self.displacement[:, :3] = self.displacement[:, -3:] = 0
        landed, masses, _ = self.dust.step(dt, self.heights_numpy)
        self.dust_loading = 0.0
        if len(self.dust.mass):
            from mars_project.environment.dust_volume import extinction_grid

            mass_density, _ = extinction_grid(self.dust.position, self.dust.mass)
            self.dust_loading = float(mass_density.max()) / self.dust.atmosphere.density
            self.peak_dust_loading = max(self.peak_dust_loading, self.dust_loading)
            if self.dust_loading > 0.1:
                raise RuntimeError(
                    f"Dust/gas mass loading {self.dust_loading:.3f} exceeds the dilute transport model's range"
                )
        if len(masses):
            p = t.as_tensor(landed, device=points.device, dtype=points.dtype)
            cell, inside = self.cell_ids(p)
            # Deposits are a static surface reservoir; they remain in the mass
            # budget, rather than disappearing when a parcel reaches the surface.
            self.deposited_height.flatten().scatter_add_(
                0,
                cell[inside],
                t.as_tensor(masses, device=points.device, dtype=points.dtype)[inside] / (DENSITY * VOXEL_SIZE**2),
            )
        self.apply_activity(dt)
        if self.surface:
            self.surface.update(self)
            self.volume_renderer.update(self.dust)


def reset_soil(env, env_ids):
    import torch

    with torch.inference_mode():
        if not hasattr(env, "soil_runtime"):
            env.soil_runtime = SoilRuntime(env)
        else:
            env.soil_runtime.reset()


def update_soil(env, env_ids):
    import torch

    if hasattr(env, "soil_runtime"):
        with torch.inference_mode():
            env.soil_runtime.update(env.step_dt)
