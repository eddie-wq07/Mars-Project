"""Check geometric sensing and the bounded, non-overlapping grain bed."""

import numpy as np
import torch

from mars_project.environment.regolith import GRAIN_RADIUS, grain_positions, surface_heights, terrain_height


def test_grain_layer_has_clearance_and_stays_in_corridor():
    p = grain_positions()
    assert 1000 < len(p) < 2000
    assert ((p[:, :2] > [-1.5, -2]) & (p[:, :2] < [5, 2])).all()
    assert np.allclose(p[:, 2] - terrain_height(p[:, 0], p[:, 1]), GRAIN_RADIUS + 0.001)
    from scipy.spatial import cKDTree

    distance, _ = cKDTree(p).query(p, k=2)
    assert distance[:, 1].min() > 2 * GRAIN_RADIUS


def test_scanner_tracks_displaced_grains_and_exposed_ground():
    hits = torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [1.0, 1.0, 0.0]]])
    centers = torch.tensor([[[0.0, 0.0, 0.03]]])
    before = surface_heights(hits, centers)
    assert torch.allclose(before, torch.tensor([[0.06, 0, 0]]))
    centers[..., 0] = 0.1
    after = surface_heights(hits, centers)
    assert torch.allclose(after, torch.tensor([[0, 0.06, 0]]))


def test_baked_grains_match_runtime_geometry():
    from pxr import Usd, UsdGeom, UsdPhysics

    from mars_project.scenes.canyon import OPTIMIZED_DIR

    stage = Usd.Stage.Open(str(OPTIMIZED_DIR / "regolith.usdc"))
    spheres = [UsdGeom.Sphere(p) for p in stage.Traverse() if p.IsA(UsdGeom.Sphere)]
    positions = grain_positions()
    assert len(spheres) == len(positions)
    for sphere, expected in zip(spheres, positions, strict=True):
        assert sphere.GetRadiusAttr().Get() == GRAIN_RADIUS
        assert sphere.GetPrim().HasAPI(UsdPhysics.RigidBodyAPI)
        assert sphere.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
        assert np.allclose(sphere.GetLocalTransformation().ExtractTranslation(), expected)
