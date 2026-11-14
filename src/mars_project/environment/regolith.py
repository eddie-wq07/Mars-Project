"""Fine-sand surface with local pressure-driven compaction and displaced rims.

A continuum approximation on 5 cm contact columns, NOT individual sand-grain DEM.
Visual vertices and height observations use physical column elevations; visuals interpolate between centers.
"""

from functools import lru_cache

import numpy as np

from mars_project.scenes.canyon import OPTIMIZED_DIR

CELL = 0.05
COLUMN_DEPTH = 0.05
X_MIN, X_MAX = -0.75, 4.25
Y_MIN, Y_MAX = -0.9, 1.2


@lru_cache(maxsize=1)
def base_grid():
    with np.load(OPTIMIZED_DIR / "local_ground.npz") as data:
        return {key: data[key] for key in data.files}


def terrain_height(x, y, grid=None):
    if grid is None:
        grid = base_grid()
    fx = np.clip((x - grid["x"][0]) / (grid["x"][1] - grid["x"][0]), 0, len(grid["x"]) - 1.000001)
    fy = np.clip((y - grid["y"][0]) / (grid["y"][1] - grid["y"][0]), 0, len(grid["y"]) - 1.000001)
    ix, iy = np.floor(fx).astype(int), np.floor(fy).astype(int)
    u, v = fx - ix, fy - iy
    h = grid["heights"]
    return np.where(
        u + v <= 1,
        h[iy, ix] + u * (h[iy, ix + 1] - h[iy, ix]) + v * (h[iy + 1, ix] - h[iy, ix]),
        h[iy + 1, ix + 1]
        + (1 - u) * (h[iy + 1, ix] - h[iy + 1, ix + 1])
        + (1 - v) * (h[iy, ix + 1] - h[iy + 1, ix + 1]),
    )


class SandField:
    """Bounded, volume-conserving surface displacement; tunable uncalibrated soil."""

    def __init__(self):
        self.x = np.arange(X_MIN + CELL / 2, X_MAX, CELL)
        self.y = np.arange(Y_MIN + CELL / 2, Y_MAX, CELL)
        self.xx, self.yy = np.meshgrid(self.x, self.y)
        self.bed = terrain_height(self.xx, self.yy)
        edge = np.minimum.reduce((self.xx - X_MIN, X_MAX - self.xx, self.yy - Y_MIN, Y_MAX - self.yy))
        self.thickness = 0.025 * np.clip(edge / 0.15, 0, 1)
        self.initial = self.bed + self.thickness
        self.heights = self.initial.copy()
        self.last_displaced = 0.0

    def reset(self):
        self.heights[:] = self.initial
        self.last_displaced = 0.0

    def sample(self, x, y):
        x, y = np.broadcast_arrays(x, y)
        ix = np.clip(np.floor((x - X_MIN) / CELL).astype(int), 0, len(self.x) - 1)
        iy = np.clip(np.floor((y - Y_MIN) / CELL).astype(int), 0, len(self.y) - 1)
        inside = (x >= X_MIN) & (x < X_MAX) & (y >= Y_MIN) & (y < Y_MAX)
        return np.where(inside, self.heights[iy, ix], terrain_height(x, y))

    def deform(self, feet, forces, yaws, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Soil timestep must be positive and finite")
        self.last_displaced = 0.0
        for foot, force, yaw in zip(feet, forces, yaws, strict=True):
            load = max(0.0, float(force[2]))
            if load < 10:
                continue
            dx, dy = self.xx - foot[0], self.yy - foot[1]
            u = np.cos(yaw) * dx + np.sin(yaw) * dy
            v = -np.sin(yaw) * dx + np.cos(yaw) * dy
            radius = np.maximum((u / 0.16) ** 2, (v / 0.085) ** 2)
            footprint = np.clip(1 - radius, 0, 1)
            # Smooth, bounded indentation under load; retain the footprint on lift-off.
            remaining = np.maximum(0, self.heights - (self.initial - np.minimum(0.012, self.thickness * 0.6)))
            removal = np.minimum(remaining, 0.025 * dt * min(load / 150.0, 2.0) * footprint)
            rim = (radius > 1.2) & (radius < 4) & (self.thickness > 0.02)
            capacity = np.maximum(0, self.initial + 0.008 - self.heights) * rim
            amount = min(removal.sum(), capacity.sum())
            if amount <= 1e-12:
                continue
            removal *= amount / removal.sum()
            deposit = capacity * (amount / capacity.sum())
            self.heights += deposit - removal
            self.last_displaced += float(amount * CELL * CELL)
        return self.last_displaced

    def centers(self):
        return np.stack((self.xx, self.yy, self.heights - COLUMN_DEPTH / 2), -1).reshape(-1, 3)


def surface_geometry(field):
    """Continuous visual sheet through the physical contact-cell centers."""
    vertices = np.stack((field.xx, field.yy, field.heights), -1).reshape(-1, 3)
    ids = np.arange(field.xx.size).reshape(field.xx.shape)
    faces = np.stack((ids[:-1, :-1], ids[:-1, 1:], ids[1:, 1:], ids[1:, :-1]), -1).reshape(-1, 4)
    return vertices, faces


def surface_normals(field):
    dy, dx = np.gradient(field.heights, CELL)
    normals = np.stack((-dx, -dy, np.ones_like(dx)), -1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals.reshape(-1, 3).astype(np.float32)


def configure_regolith(cfg):
    import isaaclab.sim as sim
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
    from isaaclab.managers import EventTermCfg

    if cfg.scene.num_envs != 1:
        raise ValueError("Local deforming sand preview currently supports one environment")
    cfg.scene.filter_collisions = False
    asset = OPTIMIZED_DIR / "regolith.usdc"
    if not asset.is_file():
        raise FileNotFoundError(f"Build the sand surface first: {asset}")
    cfg.scene.regolith_asset = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Regolith", spawn=sim.UsdFileCfg(usd_path=str(asset))
    )
    cfg.scene.regolith = RigidObjectCollectionCfg(
        rigid_objects={
            f"cell_{i:04d}": RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Regolith/cell_{i:04d}",
                spawn=None,
                init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(p)),
            )
            for i, p in enumerate(SandField().centers())
        }
    )
    cfg.scene.regolith.prim_path = "/World/envs/env_.*/Regolith"
    cfg.scene.robot.init_state.pos = (0.0, 0.0, 1.075)
    cfg.events.reset_regolith = EventTermCfg(func=reset_regolith, mode="reset")
    cfg.events.deform_regolith = EventTermCfg(func=deform_regolith, mode="interval", interval_range_s=(0.0, 0.0))
    cfg.observations.policy.height_scan.func = regolith_height_scan


class SandRuntime:
    def __init__(self, env):
        from pxr import UsdGeom

        from isaaclab.sim.utils.stage import get_current_stage

        self.env = env
        self.field = SandField()
        self.columns = env.scene["regolith"]
        self.states = self.columns.data.default_body_state.torch.clone()
        self.mesh = UsdGeom.Mesh(get_current_stage().GetPrimAtPath("/World/envs/env_0/Regolith/Surface"))
        self.foot_ids, _ = env.scene["robot"].find_bodies(".*_ankle_link")
        self.contact_ids, _ = env.scene["contact_forces"].find_bodies(".*_ankle_link")
        self.initial_vertices, self.faces = surface_geometry(self.field)
        self.total_displaced = 0.0

    def sync(self):
        import torch
        from pxr import Vt

        self.states[..., 2] = torch.as_tensor(
            self.field.centers()[:, 2], device=self.states.device, dtype=self.states.dtype
        )
        self.columns.write_body_state_to_sim(self.states)
        vertices = self.initial_vertices.copy()
        vertices[:, 2] = self.field.heights.ravel()
        self.mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(vertices.reshape(-1, 3).astype(np.float32)))
        self.mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(surface_normals(self.field)))

    def update(self, feet, forces, yaws, dt):
        amount = self.field.deform(feet, forces, yaws, dt)
        if amount > 0:
            self.sync()
            self.total_displaced += amount
        return amount


def reset_regolith(env, env_ids):
    if not hasattr(env, "sand"):
        env.sand = SandRuntime(env)
    env.sand.field.reset()
    env.sand.total_displaced = 0.0
    env.sand.sync()


def deform_regolith(env, env_ids):
    if not hasattr(env, "sand") or env.episode_length_buf[0] == 0:
        return
    robot = env.scene["robot"].data
    pose = robot.body_link_pose_w.torch[0, env.sand.foot_ids].detach().cpu().numpy()
    force = env.scene["contact_forces"].data.net_forces_w.torch[0, env.sand.contact_ids].detach().cpu().numpy()
    q = pose[:, 3:7]  # xyzw
    yaw = np.arctan2(2 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]), 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2))
    env.sand.update(pose[:, :3], force, yaw, env.step_dt)


def regolith_height_scan(env, sensor_cfg, offset=0.5):
    import torch

    sensor = env.scene[sensor_cfg.name]
    hits = sensor.data.ray_hits_w.torch
    # Read actual column poses, not a separate rendering-only deformation map.
    ix = torch.floor((hits[..., 0] - X_MIN) / CELL).long()
    iy = torch.floor((hits[..., 1] - Y_MIN) / CELL).long()
    field = env.sand.field if hasattr(env, "sand") else SandField()
    inside = (ix >= 0) & (ix < len(field.x)) & (iy >= 0) & (iy < len(field.y))
    indices = iy.clamp(0, len(field.y) - 1) * len(field.x) + ix.clamp(0, len(field.x) - 1)
    tops = env.scene["regolith"].data.body_link_pos_w.torch[..., 2] + COLUMN_DEPTH / 2
    heights = torch.where(inside, tops.gather(1, indices), hits[..., 2])
    return sensor.data.pos_w.torch[:, 2, None] - heights - offset
