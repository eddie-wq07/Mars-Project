"""A stitched terrain skin driven by MPM, never a rendered layer of solver balls."""

import numpy as np

from mars_project.scenes.canyon import OPTIMIZED_DIR


def refine_terrain(points, faces, uv, bounds, spacing=0.04):
    """Replace complete source grid cells; keep their exact boundary and UV mapping.

    The packaged canyon is a triangular regular grid. Refinement interpolates its
    actual triangles, including diagonal creases, rather than inventing a flat apron.
    """
    xs, ys = np.unique(points[:, 0]), np.unique(points[:, 1])
    xmin, ymin, xmax, ymax = bounds
    xmin, xmax = xs[xs <= xmin][-1], xs[xs >= xmax][0]
    ymin, ymax = ys[ys <= ymin][-1], ys[ys >= ymax][0]
    triangles = points[faces]
    centers = triangles.mean(1)
    remove = (centers[:, 0] > xmin) & (centers[:, 0] < xmax)
    remove &= (centers[:, 1] > ymin) & (centers[:, 1] < ymax)

    # Include every original grid line to preserve diagonal/cell junction heights.
    def subdivisions(axis, lo, hi):
        knots = axis[(axis >= lo) & (axis <= hi)]
        return np.concatenate(
            [
                np.linspace(a, b, int(np.ceil((b - a) / spacing)), endpoint=False)
                for a, b in zip(knots[:-1], knots[1:], strict=True)
            ]
            + [knots[-1:]]
        )

    x, y = subdivisions(xs, xmin, xmax), subdivisions(ys, ymin, ymax)
    xx, yy = np.meshgrid(x, y)
    vertices = np.stack((xx.ravel(), yy.ravel(), np.full(xx.size, -np.inf)), -1)
    texcoords = np.zeros((len(vertices), 2))
    for triangle, t_uv in zip(triangles[remove], uv[faces[remove]], strict=True):
        a, b, c = triangle
        u, v = b - a, c - a
        det = u[0] * v[1] - u[1] * v[0]
        if abs(det) < 1e-10:
            continue
        dx, dy = vertices[:, 0] - a[0], vertices[:, 1] - a[1]
        w1, w2 = (dx * v[1] - dy * v[0]) / det, (u[0] * dy - u[1] * dx) / det
        inside = (w1 >= -1e-6) & (w2 >= -1e-6) & (w1 + w2 <= 1 + 1e-6)
        z = a[2] + w1 * u[2] + w2 * v[2]
        inside &= z >= vertices[:, 2]
        vertices[inside, 2] = z[inside]
        texcoords[inside] = t_uv[0] + w1[inside, None] * (t_uv[1] - t_uv[0]) + w2[inside, None] * (t_uv[2] - t_uv[0])
    if not np.isfinite(vertices).all():
        raise RuntimeError("Terrain refinement left uncovered vertices")
    ids = np.arange(xx.size).reshape(xx.shape)
    a, b, c, d = ids[:-1, :-1], ids[:-1, 1:], ids[1:, :-1], ids[1:, 1:]
    local_faces = np.concatenate((np.stack((a, b, c), -1).reshape(-1, 3), np.stack((b, d, c), -1).reshape(-1, 3)))
    return faces[~remove], vertices.astype(np.float32), local_faces, texcoords.astype(np.float32), xx.shape


class SoilSurface:
    def __init__(self, env):
        import torch
        from pxr import Sdf, UsdGeom, UsdShade, Vt

        from mars_project.environment.mpm_soil import X_MAX, X_MIN, Y_MAX, Y_MIN

        stage = env.sim.stage
        terrain = UsdGeom.Mesh(stage.GetPrimAtPath("/World/MarsLabScenery/Terrain"))
        if not terrain:
            raise RuntimeError("Canyon terrain is missing")
        points = np.asarray(terrain.GetPointsAttr().Get())
        faces = np.asarray(terrain.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        uv = np.asarray(UsdGeom.PrimvarsAPI(terrain).GetPrimvar("st").Get())
        remaining, self.base, local_faces, local_uv, self.shape = refine_terrain(
            points, faces, uv, (X_MIN - 0.5, Y_MIN - 0.5, X_MAX + 0.5, Y_MAX + 0.5)
        )
        # Remove the old surface here. This is a replacement, not an overlapping sheet.
        terrain.GetFaceVertexCountsAttr().Set([3] * len(remaining))
        terrain.GetFaceVertexIndicesAttr().Set(remaining.ravel().tolist())
        self.mesh = UsdGeom.Mesh.Define(stage, "/World/SoilSurface")
        self.mesh.CreateSubdivisionSchemeAttr().Set("none")
        self.mesh.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(self.base))
        self.mesh.CreateFaceVertexCountsAttr().Set([3] * len(local_faces))
        self.mesh.CreateFaceVertexIndicesAttr().Set(local_faces.ravel().tolist())
        self.mesh.CreateExtentAttr().Set([self.base.min(0).tolist(), self.base.max(0).tolist()])
        UsdGeom.PrimvarsAPI(self.mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex").Set(
            Vt.Vec2fArray.FromNumpy(local_uv)
        )
        material = UsdShade.MaterialBindingAPI(terrain).ComputeBoundMaterial()[0]
        UsdShade.MaterialBindingAPI.Apply(self.mesh.GetPrim()).Bind(material)
        # One material and identical macro UVs across both meshes avoids a color seam.
        # Retain the canyon's photographic albedo; replace its coarse rock bump with
        # sub-millimetre sand normals on a physically scaled, world-aligned UV channel.
        path = str(material.GetPath())
        shader = UsdShade.Shader(stage.GetPrimAtPath(path + "/Surface"))
        shader.GetInput("roughness").DisconnectSource()
        shader.GetInput("roughness").Set(0.97)
        reader = UsdShade.Shader.Define(stage, path + "/SandUV")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("sandUV")
        normal = UsdShade.Shader(stage.GetPrimAtPath(path + "/Normal"))
        normal.GetInput("file").Set(str(OPTIMIZED_DIR / "sand_normal.png"))
        normal.GetInput("st").ConnectToSource(reader.ConnectableAPI(), "result")
        for mesh, positions in ((terrain, points), (self.mesh, self.base)):
            UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("sandUV", Sdf.ValueTypeNames.TexCoord2fArray, "vertex").Set(
                Vt.Vec2fArray.FromNumpy((positions[:, :2] / 0.256).astype(np.float32))
            )
        self.query = torch.as_tensor(self.base[:, :2].copy(), device=env.device)
        self.base_z = torch.as_tensor(self.base[:, 2].copy(), device=env.device)
        self.last_points = self.base.copy()

    def update(self, runtime):
        from pxr import Vt

        points = self.base.copy()
        points[:, 2] += runtime.displacement_at(self.query).cpu().numpy()
        self.mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
        # Recompute smooth geometric normals, retaining real ruts and pile-up slopes.
        grid = points.reshape(*self.shape, 3)
        dx = np.gradient(grid, axis=1)
        dy = np.gradient(grid, axis=0)
        normals = np.cross(dx, dy)
        normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12)
        self.mesh.CreateNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals.reshape(-1, 3).astype(np.float32)))
        self.mesh.SetNormalsInterpolation("vertex")
        self.mesh.GetExtentAttr().Set([points.min(0).tolist(), points.max(0).tolist()])
        self.last_points = points
