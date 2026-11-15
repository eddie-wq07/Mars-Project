"""Rebuild the fine sand normal texture used by the active MPM terrain surface."""

import numpy as np
from PIL import Image

from mars_project.scenes.canyon import OPTIMIZED_DIR


def build():
    noise = np.random.default_rng(42).normal(0, 1, (1024, 1024))
    # A 0.256 m tile: 0.25 mm per texel, filtered into a fine granular finish.
    gx, gy = np.gradient(noise)
    normal = np.stack((-0.10 * gx, -0.10 * gy, np.ones_like(noise)), -1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    Image.fromarray(np.clip((normal * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)).save(
        OPTIMIZED_DIR / "sand_normal.png"
    )


if __name__ == "__main__":
    build()
