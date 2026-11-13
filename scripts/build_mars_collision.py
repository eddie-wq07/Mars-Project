"""Crop the packaged Mars scenery into a fine, bounded walking corridor.

The source is coarse: sampling preserves its triangles, not invented microterrain.
Run with .venv/bin/python scripts/build_mars_collision.py.
"""

import json

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

from mars_project.scenes.canyon import OPTIMIZED_DIR


def build():
    source = Usd.Stage.Open(str(OPTIMIZED_DIR / "scenery.usdc"))
    mesh = UsdGeom.Mesh(source.GetPrimAtPath("/Canyon/Terrain"))
    points = np.asarray(mesh.GetPointsAttr().Get())
    assert set(mesh.GetFaceVertexCountsAttr().Get()) == {3}
    transform = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim()))
    points = points @ transform[:3, :3] + transform[3, :3]
    triangles = points[np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)]
    xs, ys = np.linspace(-1.5, 5, 131), np.linspace(-2, 2, 81)
    xx, yy = np.meshgrid(xs, ys)
    heights = np.full(xx.shape, -np.inf)
    selected = triangles[
        np.all(triangles.max(1)[:, :2] >= [-1.5, -2], 1) & np.all(triangles.min(1)[:, :2] <= [5, 2], 1)
    ]
    for a, b, c in selected:
        u, v = b - a, c - a
        det = u[0] * v[1] - u[1] * v[0]
        if abs(det) < 1e-12:
            continue
        w1 = ((xx - a[0]) * v[1] - (yy - a[1]) * v[0]) / det
        w2 = (u[0] * (yy - a[1]) - u[1] * (xx - a[0])) / det
        inside = (w1 >= -1e-7) & (w2 >= -1e-7) & (w1 + w2 <= 1 + 1e-7)
        heights = np.maximum(heights, np.where(inside, a[2] + w1 * u[2] + w2 * v[2], -np.inf))
    if not np.isfinite(heights).all():
        raise RuntimeError("Mars corridor has holes in its source surface")
    vertices = np.stack((xx, yy, heights), -1).reshape(-1, 3)
    cells = (np.arange(len(ys) - 1)[:, None] * len(xs) + np.arange(len(xs) - 1)).ravel()
    faces = np.concatenate(
        (
            np.stack((cells, cells + 1, cells + len(xs)), 1),
            np.stack((cells + 1, cells + len(xs) + 1, cells + len(xs)), 1),
        )
    )
    stage = Usd.Stage.Open(str(OPTIMIZED_DIR / "local_ground.usdc"))
    ground = UsdGeom.Mesh(stage.GetPrimAtPath("/Canyon/Ground"))
    ground.GetPointsAttr().Set(vertices.astype(np.float32).tolist())
    ground.GetFaceVertexCountsAttr().Set([3] * len(faces))
    ground.GetFaceVertexIndicesAttr().Set(faces.ravel().tolist())
    ground.CreateExtentAttr().Set([vertices.min(0).tolist(), vertices.max(0).tolist()])
    ground.CreateSubdivisionSchemeAttr().Set("none")
    ground.CreateVisibilityAttr().Set("invisible")
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim()).CreateCollisionEnabledAttr().Set(True)
    UsdPhysics.MeshCollisionAPI.Apply(ground.GetPrim()).CreateApproximationAttr().Set("none")
    material = UsdShade.Material.Define(stage, "/Canyon/GroundMaterial")
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr().Set(0.8)
    physics.CreateDynamicFrictionAttr().Set(0.6)
    physics.CreateRestitutionAttr().Set(0.0)
    UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(material, materialPurpose="physics")
    stage.GetRootLayer().Save()
    np.savez_compressed(OPTIMIZED_DIR / "local_ground.npz", x=xs, y=ys, heights=heights)
    metadata = dict(
        source="scenery.usdc:/Canyon/Terrain",
        bounds_xy=[[-1.5, -2], [5, 2]],
        spacing_m=0.05,
        vertices=len(vertices),
        triangles=len(faces),
        source_triangles=len(selected),
        route_direction="+X",
        spawn_ground_z=float(heights[40, 30]),
        limitation="Resampled source triangles; no added microterrain",
    )
    (OPTIMIZED_DIR / "local_ground.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    build()
