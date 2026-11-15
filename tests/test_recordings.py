"""Moving orbits must keep their subject on the optical axis."""

import math

import pytest
from pxr import Gf, Usd, UsdGeom

from mars_project.recordings import DollyOrbitCamera, OrbitCamera, SmoothCameraTarget


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
        optical_center = matrix.GetInverse().Transform(Gf.Vec3d(*orbit.center))
        assert optical_center[0] == pytest.approx(0, abs=1e-8)
        assert optical_center[1] == pytest.approx(0, abs=1e-8)
        assert optical_center[2] == pytest.approx(-2.6)
        assert math.dist(matrix.ExtractTranslation(), orbit.center) == pytest.approx(2.6)


def test_tracking_rejects_footstep_bounce_but_follows_travel():
    tracker = SmoothCameraTarget((0, 0, 1))
    vertical = []
    for frame in range(250):
        t = (frame + 1) / 25
        center = tracker.update((0, -0.5 * t, 1 + 0.08 * math.sin(2 * math.pi * 2 * t)), 1 / 25)
        if frame > 125:
            vertical.append(center[2])
        assert abs(center[1] + 0.5 * t) < 0.13
    # An 8 cm, 2 Hz gait bounce becomes less than 1 cm peak-to-peak.
    assert max(vertical) - min(vertical) < 0.01
    assert abs(center[2] - 1) < 0.01


def test_matched_dolly_orbit_holds_front_then_moves_viewer_right():
    for heading in (0.0, -math.pi / 2):
        stage = Usd.Stage.CreateInMemory()
        camera = UsdGeom.Camera.Define(stage, "/Camera")
        UsdGeom.Xformable(camera).AddTransformOp()
        shot = DollyOrbitCamera(stage, "/Camera", (0, 0, 1), heading)
        for progress, relative_angle in ((0, 0), (1 / 3, 0), (2 / 3, math.radians(25)), (1, math.radians(50))):
            shot.update(progress)
            matrix = UsdGeom.Xformable(camera).GetLocalTransformation()
            eye = matrix.ExtractTranslation() - Gf.Vec3d(*shot.center)
            assert tuple(eye) == pytest.approx(
                (2.6 * math.cos(heading + relative_angle), 2.6 * math.sin(heading + relative_angle), 0)
            )
            assert tuple(matrix.GetInverse().Transform(Gf.Vec3d(*shot.center))) == pytest.approx((0, 0, -2.6))
        assert camera.GetFocalLengthAttr().Get() == 22
        assert camera.GetHorizontalApertureAttr().Get() == 36
        assert camera.GetVerticalApertureAttr().Get() == 20.25
