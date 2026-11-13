"""Bake only the outward walking corridor of Verdant Trail into a static height mesh.

Run with .venv/bin/python scripts/build_earth_collision.py. No simulator needed.
Mesh_0095 is the continuous trail substrate; foliage/rocks/overhangs are excluded.
"""

import json

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

from mars_project.scenes.earth import EARTH_ASSET, USD_DIR


def build():
    stage = Usd.Stage.Open(str(EARTH_ASSET))
    instancer = UsdGeom.PointInstancer(stage.GetPrimAtPath("/Earth/VegetationAndTerrain"))
    targets = instancer.GetPrototypesRel().GetTargets()
    prototype = next(i for i, p in enumerate(targets) if p.name == "Mesh_0095")
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(targets[prototype].AppendChild("Geometry")))
    points = np.array(mesh.GetPointsAttr().Get())
    counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
    indices = np.array(mesh.GetFaceVertexIndicesAttr().Get())
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    faces = np.concatenate(
        [
            np.stack(
                (indices[starts[counts > k]], indices[starts[counts > k] + k - 1], indices[starts[counts > k] + k]), 1
            )
            for k in range(2, int(counts.max()))
        ]
    )
    anchor = np.array(json.loads((USD_DIR / "conversion.json").read_text())["anchor"])
    transforms = np.array(instancer.ComputeInstanceTransformsAtTime(Usd.TimeCode.Default(), Usd.TimeCode.Default()))
    # Includes foot placement and the policy's 1.6 x 1 m height scanner margin.
    xs = np.linspace(-2, 2, 81)
    ys = np.linspace(-5, 1.5, 131)
    xx, yy = np.meshgrid(xs, ys)
    heights = np.full(xx.shape, -np.inf)
    for instance in np.where(np.array(instancer.GetProtoIndicesAttr().Get()) == prototype)[0]:
        transform = transforms[instance]
        world = points @ transform[:3, :3] + transform[3, :3] - anchor
        triangles = world[faces]
        triangles = triangles[np.all(triangles.max(1) >= [-2, -5, -1], 1) & np.all(triangles.min(1) <= [2, 1.5, 1], 1)]
        for a, b, c in triangles:
            u, v = b - a, c - a
            det = u[0] * v[1] - u[1] * v[0]
            if abs(det) < 1e-12:
                continue
            ix = np.flatnonzero((xs >= min(a[0], b[0], c[0]) - 1e-7) & (xs <= max(a[0], b[0], c[0]) + 1e-7))
            iy = np.flatnonzero((ys >= min(a[1], b[1], c[1]) - 1e-7) & (ys <= max(a[1], b[1], c[1]) + 1e-7))
            region = np.ix_(iy, ix)
            dx, dy = xx[region] - a[0], yy[region] - a[1]
            w1, w2 = (dx * v[1] - dy * v[0]) / det, (u[0] * dy - u[1] * dx) / det
            inside = (w1 >= -1e-6) & (w2 >= -1e-6) & (w1 + w2 <= 1 + 1e-6)
            heights[region] = np.maximum(heights[region], np.where(inside, a[2] + w1 * u[2] + w2 * v[2], -np.inf))
    if not np.isfinite(heights).all():
        raise RuntimeError("Trail crop has holes; do not interpolate across missing source geometry")
    vertices = np.stack((xx, yy, heights), -1).reshape(-1, 3)
    cells = (np.arange(len(ys) - 1)[:, None] * len(xs) + np.arange(len(xs) - 1)).ravel()
    faces = np.concatenate(
        (
            np.stack((cells, cells + 1, cells + len(xs)), 1),
            np.stack((cells + 1, cells + len(xs) + 1, cells + len(xs)), 1),
        )
    )
    path = USD_DIR / "earth_collision_ground.usda"
    output = Usd.Stage.CreateNew(str(path)) if not path.exists() else Usd.Stage.Open(str(path))
    root = output.DefinePrim("/CollisionGround", "Xform")
    output.SetDefaultPrim(root)
    UsdGeom.SetStageUpAxis(output, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(output, 1.0)
    ground = UsdGeom.Mesh.Define(output, "/CollisionGround/Plane")
    ground.CreateVisibilityAttr().Set("invisible")
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim()).CreateCollisionEnabledAttr().Set(True)
    material = UsdShade.Material.Define(output, "/CollisionGround/PhysicsMaterial")
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr().Set(1.0)
    physics_material.CreateDynamicFrictionAttr().Set(1.0)
    physics_material.CreateRestitutionAttr().Set(0.0)
    UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(material, materialPurpose="physics")
    ground.GetPointsAttr().Set(vertices.astype(np.float32).tolist())
    ground.GetFaceVertexCountsAttr().Set([3] * len(faces))
    ground.GetFaceVertexIndicesAttr().Set(faces.ravel().tolist())
    ground.CreateExtentAttr().Set([vertices.min(0).tolist(), vertices.max(0).tolist()])
    ground.CreateSubdivisionSchemeAttr().Set("none")
    UsdPhysics.MeshCollisionAPI.Apply(ground.GetPrim()).CreateApproximationAttr().Set("none")
    output.GetRootLayer().Save()
    metadata = dict(
        source_prototype=str(targets[prototype]),
        bounds_xy=[[-2, -5], [2, 1.5]],
        spacing_m=0.05,
        vertices=len(vertices),
        triangles=len(faces),
        height_range_m=[float(heights.min()), float(heights.max())],
        spawn_ground_z=float(heights[np.argmin(abs(ys)), np.argmin(abs(xs))]),
    )
    (USD_DIR / "earth_collision_ground.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    build()
