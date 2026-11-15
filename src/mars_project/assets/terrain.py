"""Sample the packaged canyon collision mesh without launching the simulator."""

from functools import lru_cache

import numpy as np

from mars_project.assets.scene_paths import GRAND_CANYON_DIR


@lru_cache(maxsize=1)
def base_grid():
    with np.load(GRAND_CANYON_DIR / "optimized" / "local_ground.npz") as data:
        return {key: data[key] for key in data.files}


def terrain_height(x, y, grid=None):
    """Interpolate the grid's triangles, clamping queries to its outer boundary."""
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
