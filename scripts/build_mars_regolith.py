"""Rebuild the retired heuristic's assets; these are not used by playback."""

import numpy as np
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

from mars_project.environment.regolith import CELL, COLUMN_DEPTH, SandField, surface_geometry, surface_normals
from mars_project.scenes.canyon import OPTIMIZED_DIR


def build():
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 1, (1024, 1024))
    # A 0.256 m tile: 0.25 mm per texel, filtered into a fine granular finish.
    rgb = np.clip(np.array([80, 54, 35]) + noise[:, :, None] * 5, 0, 255).astype(np.uint8)
    Image.fromarray(rgb).save(OPTIMIZED_DIR / "sand_albedo.png")
    gx, gy = np.gradient(noise)
    normal = np.stack((-0.10 * gx, -0.10 * gy, np.ones_like(noise)), -1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    Image.fromarray(np.clip((normal * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)).save(
        OPTIMIZED_DIR / "sand_normal.png"
    )
    stage = Usd.Stage.CreateNew(str(OPTIMIZED_DIR / "regolith.usdc"))
    root = stage.DefinePrim("/Regolith", "Xform")
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    material = UsdShade.Material.Define(stage, "/Regolith/Material")
    shader = UsdShade.Shader.Define(stage, "/Regolith/Material/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.95)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    reader = UsdShade.Shader.Define(stage, "/Regolith/Material/UV")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    for name, file, color_space in [("Color", "sand_albedo.png", "sRGB"), ("Normal", "sand_normal.png", "raw")]:
        tex = UsdShade.Shader.Define(stage, "/Regolith/Material/" + name)
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("./" + file)
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(color_space)
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        if name == "Normal":
            tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set((2, 2, 2, 1))
            tex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set((-1, -1, -1, 0))
        shader.CreateInput(
            "diffuseColor" if name == "Color" else "normal",
            Sdf.ValueTypeNames.Color3f if name == "Color" else Sdf.ValueTypeNames.Normal3f,
        ).ConnectToSource(tex.ConnectableAPI(), "rgb")
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr().Set(0.65)
    physics.CreateDynamicFrictionAttr().Set(0.45)
    physics.CreateRestitutionAttr().Set(0.0)
    field = SandField()
    for i, pos in enumerate(field.centers()):
        cube = UsdGeom.Cube.Define(stage, f"/Regolith/cell_{i:04d}")
        cube.CreateSizeAttr().Set(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*pos))
        cube.AddScaleOp().Set(Gf.Vec3f(CELL, CELL, COLUMN_DEPTH))
        cube.CreateVisibilityAttr().Set("invisible")
        prim = cube.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr().Set(True)
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
        prim.AddAppliedSchema("PhysxCollisionAPI")
        prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.001)
        prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)
    points, faces = surface_geometry(field)
    mesh = UsdGeom.Mesh.Define(stage, "/Regolith/Surface")
    mesh.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr().Set([4] * len(faces))
    mesh.CreateFaceVertexIndicesAttr().Set(faces.ravel().tolist())
    mesh.CreateSubdivisionSchemeAttr().Set("none")
    mesh.CreateNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(surface_normals(field)))
    mesh.SetNormalsInterpolation("vertex")
    mesh.CreateExtentAttr().Set([points.min(0).tolist(), points.max(0).tolist()])
    uv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex")
    uv.Set(Vt.Vec2fArray.FromNumpy((points[:, :2] / 0.256).astype(np.float32)))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()
    print(f"SAND_BUILT cells={field.xx.size} collision_spacing={CELL} visual_grain_texel=0.00025")


if __name__ == "__main__":
    build()
