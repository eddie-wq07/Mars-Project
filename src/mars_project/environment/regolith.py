"""Coarse, physically movable grains over the local rigid Mars surface.

This shallow rigid-grain bed is a contact approximation, not calibrated fine soil.
"""

import numpy as np

from mars_project.scenes.canyon import OPTIMIZED_DIR

GRAIN_RADIUS = 0.03
GRAIN_DENSITY = 1600.0


def grain_positions():
    """A single non-overlapping layer in the foot-contact band, not the scenery."""
    grid = np.load(OPTIMIZED_DIR / "local_ground.npz")
    spacing = 2 * GRAIN_RADIUS + 0.002
    x = np.arange(-0.6, 4.4, spacing)
    y = np.arange(-0.65, 0.66, spacing)
    xx, yy = np.meshgrid(x, y)
    # Exact interpolation on the builder's same two-triangle grid.
    z = terrain_height(xx, yy, grid)
    return np.stack((xx, yy, z + GRAIN_RADIUS + 0.001), -1).reshape(-1, 3)


def terrain_height(x, y, grid=None):
    if grid is None:
        grid = np.load(OPTIMIZED_DIR / "local_ground.npz")
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


def configure_regolith(cfg):
    import isaaclab.sim as sim
    from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
    from isaaclab.managers import EventTermCfg

    # The preview uses one environment; cross-environment filtering is unnecessary
    # and scales poorly with the many individually movable grain assets.
    if cfg.scene.num_envs == 1:
        cfg.scene.filter_collisions = False
    # Spawn the bed as ONE asset so the scene cloner does not plan thousands
    # of independent assets. The collection still exposes each movable body.
    from isaaclab.assets import AssetBaseCfg

    asset = OPTIMIZED_DIR / "regolith.usdc"
    if not asset.is_file():
        raise FileNotFoundError(f"Build the grain layer first: {asset}")
    cfg.scene.regolith_asset = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Regolith",
        spawn=sim.UsdFileCfg(usd_path=str(asset)),
    )
    cfg.scene.regolith = RigidObjectCollectionCfg(
        rigid_objects={
            f"grain_{i:04d}": RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Regolith/grain_{i:04d}",
                spawn=None,
                init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(p)),
            )
            for i, p in enumerate(grain_positions())
        }
    )
    # The scene sorts entities by prim_path depth. Give the collection its
    # parent path so the preceding bed asset spawns before its body handles.
    cfg.scene.regolith.prim_path = "/World/envs/env_.*/Regolith"
    cfg.scene.robot.init_state.pos = (0.0, 0.0, 1.05 + 2 * GRAIN_RADIUS)
    cfg.events.reset_regolith = EventTermCfg(func=reset_regolith, mode="reset")
    cfg.observations.policy.height_scan.func = regolith_height_scan


def reset_regolith(env, env_ids):
    grains = env.scene["regolith"]
    states = grains.data.default_body_state.torch[env_ids].clone()
    states[..., :3] += env.scene.env_origins[env_ids, None, :]
    grains.write_body_state_to_sim(states, env_ids=env_ids)
    grains.reset(env_ids)


def surface_heights(hits, centers, radius=GRAIN_RADIUS):
    """Vertical ray/sphere intersections, combined with static ground hits.

    Uses live PhysX positions. Chunking caps temporary memory as grain count grows.
    """
    import torch

    heights = hits[..., 2].clone()
    for chunk in centers.split(256, dim=1):
        d2 = ((hits[:, :, None, :2] - chunk[:, None, :, :2]) ** 2).sum(-1)
        top = chunk[:, None, :, 2] + torch.sqrt(torch.clamp(radius * radius - d2, min=0))
        top = torch.where(d2 <= radius * radius, top, -torch.inf)
        heights = torch.maximum(heights, top.amax(-1))
    return heights


def regolith_height_scan(env, sensor_cfg, offset=0.5):
    sensor = env.scene[sensor_cfg.name]
    heights = surface_heights(sensor.data.ray_hits_w.torch, env.scene["regolith"].data.body_link_pos_w.torch)
    return sensor.data.pos_w.torch[:, 2, None] - heights - offset
