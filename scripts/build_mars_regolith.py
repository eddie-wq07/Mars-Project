"""Bake the local grain bed as one spawn asset with independent rigid grains."""

import math

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from mars_project.environment.regolith import GRAIN_DENSITY, GRAIN_RADIUS, grain_positions
from mars_project.scenes.canyon import OPTIMIZED_DIR


def build():
    path = OPTIMIZED_DIR / "regolith.usdc"
    stage = Usd.Stage.CreateNew(str(path))
    root = stage.DefinePrim("/Regolith", "Xform")
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    material = UsdShade.Material.Define(stage, "/Regolith/Material")
    shader = UsdShade.Shader.Define(stage, "/Regolith/Material/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.32, 0.16, 0.07))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr().Set(0.9)
    physics.CreateDynamicFrictionAttr().Set(0.7)
    physics.CreateRestitutionAttr().Set(0.0)
    for i, pos in enumerate(grain_positions()):
        sphere = UsdGeom.Sphere.Define(stage, f"/Regolith/grain_{i:04d}")
        sphere.CreateRadiusAttr().Set(GRAIN_RADIUS)
        sphere.AddTranslateOp().Set(Gf.Vec3d(*pos))
        prim = sphere.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(GRAIN_DENSITY * 4 / 3 * math.pi * GRAIN_RADIUS**3)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
        UsdShade.MaterialBindingAPI(prim).Bind(material, materialPurpose="physics")
        prim.AddAppliedSchema("PhysxRigidBodyAPI")
        prim.AddAppliedSchema("PhysxCollisionAPI")
        for name, value in [
            ("physxRigidBody:linearDamping", 0.0),
            ("physxRigidBody:angularDamping", 10.0),
            ("physxRigidBody:maxDepenetrationVelocity", 0.5),
            ("physxCollision:contactOffset", 0.002),
            ("physxCollision:restOffset", 0.0),
        ]:
            prim.CreateAttribute(name, Sdf.ValueTypeNames.Float).Set(value)
        for name, value in [("solverPositionIterationCount", 8), ("solverVelocityIterationCount", 2)]:
            prim.CreateAttribute("physxRigidBody:" + name, Sdf.ValueTypeNames.Int).Set(value)
    stage.GetRootLayer().Save()
    print(f"REGOLITH_BUILT grains={len(grain_positions())} radius={GRAIN_RADIUS} path={path}")


if __name__ == "__main__":
    build()
