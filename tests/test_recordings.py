"""Moving orbits must keep their subject on the optical axis."""

import math

import pytest
from pxr import Gf, Usd, UsdGeom

from mars_project.recordings import OrbitCamera


def test_orbit_tracks_moving_midpoint_without_changing_radius():
    stage = Usd.Stage.CreateInMemory()
    camera = UsdGeom.Camera.Define(stage, "/Camera")
    UsdGeom.Xformable(camera).AddTransformOp().Set(
        Gf.Matrix4d().SetLookAt(Gf.Vec3d(0, -6, 1), Gf.Vec3d(0, 0, 1), Gf.Vec3d(0, 0, 1)).GetInverse()
    )
    orbit = OrbitCamera(stage, "/Camera", (0, 0, 1))
    for progress in (0, 0.25, 0.5, 0.75, 1):
        center = Gf.Vec3d(0.2 * progress, -3 * progress, 1 - 0.1 * progress)
        orbit.update(progress, center)
        matrix = UsdGeom.Xformable(camera).GetLocalTransformation()
        optical_center = matrix.GetInverse().Transform(center)
        assert optical_center[0] == pytest.approx(0, abs=1e-8)
        assert optical_center[1] == pytest.approx(0, abs=1e-8)
        assert optical_center[2] == pytest.approx(-2.6)
        assert math.dist(matrix.ExtractTranslation(), center) == pytest.approx(2.6)
