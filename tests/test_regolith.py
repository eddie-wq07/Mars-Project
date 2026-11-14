"""Legacy surface bookkeeping checks; these do not validate soil mechanics."""

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics

from mars_project.environment.regolith import CELL, COLUMN_DEPTH, SandField, surface_geometry
from mars_project.scenes.canyon import OPTIMIZED_DIR


def test_pressure_leaves_local_footprint_and_displaced_rim():
    f = SandField()
    feet = np.array([[0.4, 0, 0.07]])
    force = np.array([[0, 0, 120.0]])
    f.deform(feet, force * 0, [0], 0.02)
    assert np.array_equal(f.heights, f.initial)
    for _ in range(50):
        f.deform(feet, force, [0], 0.02)
    delta = f.heights - f.initial
    assert delta.min() < -0.005 and delta.max() > 0
    assert abs(delta.sum() * CELL * CELL) < 1e-12
    assert delta.min() >= -0.012 - 1e-12
    assert np.all(delta[f.xx > 1] == 0)
    assert f.sample(np.array([0.4]), np.array([0])) < 0.025
    f.reset()
    assert np.array_equal(f.heights, f.initial)


def test_surface_vertices_match_contact_column_tops():
    f = SandField()
    f.deform(np.array([[0.4, 0, 0.07]]), np.array([[0, 0, 150.0]]), [0], 0.1)
    points, faces = surface_geometry(f)
    tops = points[:, 2]
    assert np.allclose(tops, f.heights.ravel())
    assert np.allclose(f.centers()[:, 2] + COLUMN_DEPTH / 2, f.heights.ravel())
    assert faces.max() < len(points)


def test_baked_sand_has_no_visible_balls():
    stage = Usd.Stage.Open(str(OPTIMIZED_DIR / "regolith.usdc"))
    assert not any(p.IsA(UsdGeom.Sphere) for p in stage.Traverse())
    cells = [p for p in stage.Traverse() if p.IsA(UsdGeom.Cube)]
    assert len(cells) == SandField().xx.size
    for p in cells:
        assert UsdGeom.Imageable(p).GetVisibilityAttr().Get() == "invisible"
        assert UsdPhysics.RigidBodyAPI(p).GetKinematicEnabledAttr().Get()
        assert p.HasAPI(UsdPhysics.CollisionAPI)
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Regolith/Surface"))
    assert len(mesh.GetPointsAttr().Get()) == len(cells)
