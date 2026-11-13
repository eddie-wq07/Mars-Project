"""The local Mars mesh must cover the short route and match the scenery."""

import numpy as np
import pytest
from pxr import Usd, UsdGeom

from mars_project.scenes.canyon import OPTIMIZED_DIR


def test_local_corridor_matches_source_and_covers_scanner():
    stage = Usd.Stage.Open(str(OPTIMIZED_DIR / "local_ground.usdc"))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Canyon/Ground"))
    points = np.asarray(mesh.GetPointsAttr().Get())
    assert np.isfinite(points).all()
    assert np.allclose(points.min(0)[:2], [-1.5, -2])
    assert np.allclose(points.max(0)[:2], [5, 2])
    assert set(mesh.GetFaceVertexCountsAttr().Get()) == {3}
    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    t = points[faces]
    signed_area = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])[:, 2] / 2
    assert (signed_area > 0).all()
    assert signed_area.sum() == pytest.approx(26)
    source = Usd.Stage.Open(str(OPTIMIZED_DIR / "scenery.usdc"))
    m = UsdGeom.Mesh(source.GetPrimAtPath("/Canyon/Terrain"))
    tri = np.asarray(m.GetPointsAttr().Get())[np.asarray(m.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)]
    tri = tri[(tri.max(1)[:, 0] >= -1.5) & (tri.min(1)[:, 0] <= 5) & (tri.max(1)[:, 1] >= -2) & (tri.min(1)[:, 1] <= 2)]
    for p in points[::101]:
        heights = []
        for a, b, c in tri:
            uv = np.linalg.solve(np.stack((b[:2] - a[:2], c[:2] - a[:2]), axis=1), p[:2] - a[:2])
            if min(uv) >= -1e-5 and sum(uv) <= 1 + 1e-5:
                heights.append(a[2] + uv[0] * (b[2] - a[2]) + uv[1] * (c[2] - a[2]))
        assert heights and p[2] == pytest.approx(max(heights), abs=1e-6)
