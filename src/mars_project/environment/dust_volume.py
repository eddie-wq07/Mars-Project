"""Render parcel extinction as a heterogeneous volume, never enlarged opaque grains."""

import tempfile
from pathlib import Path

import numpy as np


def extinction_grid(positions, cross_sections, spacing=0.025):
    """Cloud-in-cell projection, preserving integrated extinction area [m²]."""
    lower = np.floor(positions.min(0) / spacing) * spacing - 2 * spacing
    shape = np.ceil((positions.max(0) - lower) / spacing).astype(int) + 4
    if np.prod(shape) > 8_000_000:
        raise RuntimeError("Dust render domain exceeded its explicit memory budget")
    field = np.zeros(shape, dtype=np.float32)
    coord = (positions - lower) / spacing
    index = np.floor(coord).astype(int)
    fraction = coord - index
    for x in (0, 1):
        for y in (0, 1):
            for z in (0, 1):
                delta = np.array([x, y, z])
                weight = np.prod(np.where(delta, fraction, 1 - fraction), axis=1)
                np.add.at(field, tuple((index + delta).T), cross_sections * weight / spacing**3)
    # Normalized compact filter changes reconstruction width, not optical mass.
    padded = np.pad(field, 1)
    result = np.zeros_like(field)
    for x, wx in enumerate((0.25, 0.5, 0.25)):
        for y, wy in enumerate((0.25, 0.5, 0.25)):
            for z, wz in enumerate((0.25, 0.5, 0.25)):
                result += wx * wy * wz * padded[x : x + shape[0], y : y + shape[1], z : z + shape[2]]
    return result, lower


class DustVolume:
    def __init__(self, env):
        import carb.settings
        from pxr import Sdf, UsdGeom, UsdShade

        self.env = env
        self.cache = tempfile.TemporaryDirectory(prefix="mars-dust-volume-")
        self.frame = 0
        stage = env.sim.stage
        self.box = UsdGeom.Mesh.Define(stage, "/World/AirborneDust")
        self.box.CreateSubdivisionSchemeAttr().Set("none")
        self.box.CreateFaceVertexCountsAttr().Set([4] * 6)
        self.box.CreateFaceVertexIndicesAttr().Set(
            [0, 2, 3, 1, 4, 5, 7, 6, 0, 1, 5, 4, 2, 6, 7, 3, 0, 4, 6, 2, 1, 3, 7, 5]
        )
        self.box.CreateVisibilityAttr().Set("invisible")
        self.box.GetPrim().CreateAttribute("primvars:isVolume", Sdf.ValueTypeNames.Bool).Set(True)
        material = UsdShade.Material.Define(stage, "/World/AirborneDustMaterial")
        shader = UsdShade.Shader.Define(stage, "/World/AirborneDustMaterial/Shader")
        shader.CreateImplementationSourceAttr().Set("sourceAsset")
        shader.SetSourceAsset("OmniVolumeDensity.mdl", "mdl")
        shader.SetSourceAssetSubIdentifier("OmniVolumeDensity", "mdl")
        shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
        material.CreateVolumeOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
        self.texture = shader.CreateInput("volume_density_texture", Sdf.ValueTypeNames.Asset)
        self.texture.SetRenderType("texture_3d")
        shader.CreateInput("volume_density_scale", Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateInput("volume_albedo", Sdf.ValueTypeNames.Color3f).Set((0.8, 0.6, 0.4))
        shader.CreateInput("directional_bias", Sdf.ValueTypeNames.Float).Set(0.6)
        UsdShade.MaterialBindingAPI.Apply(self.box.GetPrim()).Bind(material)
        carb.settings.get_settings().set("/rtx/pathtracing/ptvol/enabled", True)

    def update(self, cloud):
        import omni.volume
        from pxr import Vt

        if not len(cloud.mass):
            self.box.GetVisibilityAttr().Set("invisible")
            return
        field, lower = extinction_grid(cloud.position, cloud.extinction_area())
        spacing = 0.025
        # Unique files prevent stale renderer texture caches; TemporaryDirectory
        # owns them for this process only and never overwrites user artifacts.
        path = Path(self.cache.name) / f"dust-{self.frame:06d}.nvdb"
        self.frame += 1
        # Use Kit's own writer so the volume version matches its renderer.
        # Its dense-array order is z,y,x; our conservative field is x,y,z.
        interface = omni.volume.get_volume_interface()
        volume = interface.create_from_dense(
            0.0, np.ascontiguousarray(field.transpose(2, 1, 0)), spacing, lower.astype(np.float64), "density"
        )
        if not interface.save_volume(volume, str(path)):
            raise RuntimeError("Kit failed to write the dust density volume")
        upper = lower + np.asarray(field.shape) * spacing
        corners = np.array(
            [[x, y, z] for z in (lower[2], upper[2]) for y in (lower[1], upper[1]) for x in (lower[0], upper[0])],
            dtype=np.float32,
        )
        self.box.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(corners))
        self.box.CreateExtentAttr().Set([lower.tolist(), upper.tolist()])
        self.texture.Set(str(path))
        self.box.GetVisibilityAttr().Set("inherited")
