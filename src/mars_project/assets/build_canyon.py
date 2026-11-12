"""Offline canyon preparation: bounded textures and a matching local collider.

Run with the project Python: python -m mars_project.assets.build_canyon
The original downloaded USDZ and preview layer remain untouched.
"""
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
from scipy.ndimage import gaussian_filter
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

from mars_project.assets.scene_paths import GRAND_CANYON_DIR


def build_sky():
    """Small latlong daytime sky, with a warm horizon and no volume simulation."""
    output = GRAND_CANYON_DIR / 'optimized'
    output.mkdir(exist_ok=True)
    latitude = np.linspace(np.pi / 2, -np.pi / 2, 512)[:, None, None]
    altitude = np.maximum(np.sin(latitude), 0) ** .55
    horizon = np.array([.90, .72, .55])[None, None, :]
    zenith = np.array([.43, .49, .59])[None, None, :]
    sky = horizon * (1 - altitude) + zenith * altitude
    # Darker lower hemisphere avoids lighting terrain as if the sky were below it.
    sky = np.where(latitude < 0, np.array([.20, .15, .11])[None, None, :], sky)
    sky = np.broadcast_to(sky, (512, 1024, 3))
    Image.fromarray(np.uint8(np.clip(sky, 0, 1) * 255)).save(output / 'sky.png')


def build():
    folder = GRAND_CANYON_DIR
    output = folder / 'optimized'
    output.mkdir(exist_ok=True)
    build_sky()
    # This is the known downloaded 20k source, decoded offline in CPU RAM only.
    Image.MAX_IMAGE_PIXELS = 450_000_000
    with zipfile.ZipFile(folder / 'grand_canyon.usdz') as archive:
        name = next(n for n in archive.namelist() if n.endswith('terrain_albedo_mars_enhanced.png'))
        with archive.open(name) as stream:
            image = Image.open(stream).convert('RGB')
            image.thumbnail((4096, 4096), Image.Resampling.LANCZOS)
            ImageEnhance.Color(image).enhance(0.8).save(output / 'albedo.jpg', quality=93)
    rng = np.random.default_rng(7)
    noise = rng.normal(size=(512, 512))
    height = sum(gaussian_filter(noise, sigma, mode='wrap') * weight
                 for sigma, weight in ((1, .4), (4, 1), (16, 2)))
    dy, dx = np.gradient(height)
    normals = np.stack((-dx * 1.5, -dy * 1.5, np.ones_like(height)), axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    Image.fromarray(np.uint8(np.clip(normals * .5 + .5, 0, 1) * 255)).save(output / 'rock_normal.png')
    rough = np.clip(.83 + height * .12, .65, .98)
    Image.fromarray(np.uint8(rough * 255)).save(output / 'roughness.png')
    source = Usd.Stage.Open(str(folder / 'grand_canyon_preview.usdc'))
    original = UsdGeom.Mesh(source.GetPrimAtPath('/World/MarsTerrain/VisualMesh'))
    points = np.array(original.GetPointsAttr().Get(), dtype=np.float32)
    anchor = json.loads((folder / 'preview-anchor.json').read_text())
    points -= np.array([anchor['x'], anchor['y'], anchor['z']], dtype=np.float32)
    radius = np.linalg.norm(points[:, :2], axis=1)
    blend = np.clip((radius - 9) / 12, 0, 1)
    blend = blend * blend * (3 - 2 * blend)
    small = .025 * np.sin(points[:, 0] * .6) * np.cos(points[:, 1] * .5)
    points[:, 2] = points[:, 2] * blend + small * (1 - blend)
    faces = np.array(original.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3)
    uv = UsdGeom.PrimvarsAPI(original).GetPrimvar('st')
    vertex_normals = np.zeros_like(points)
    tri = points[faces]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for column in range(3):
        np.add.at(vertex_normals, faces[:, column], face_normals)
    vertex_normals /= np.maximum(np.linalg.norm(vertex_normals, axis=1, keepdims=True), 1e-8)

    def stage_at(name):
        path = output / name
        if path.exists():
            path.unlink()
        stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageUpAxis(stage, 'Z')
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        root = UsdGeom.Xform.Define(stage, '/Canyon')
        stage.SetDefaultPrim(root.GetPrim())
        return stage

    stage = stage_at('scenery.usdc')
    mesh = UsdGeom.Mesh.Define(stage, '/Canyon/Terrain')
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.ravel()))
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(vertex_normals))
    mesh.SetNormalsInterpolation('vertex')
    new_uv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', uv.GetTypeName(), uv.GetInterpolation())
    new_uv.Set(uv.Get())
    if uv.IsIndexed():
        new_uv.SetIndices(uv.GetIndices())
    material = UsdShade.Material.Define(stage, '/Canyon/Rock')
    surface = UsdShade.Shader.Define(stage, '/Canyon/Rock/Surface')
    surface.CreateIdAttr('UsdPreviewSurface')
    surface.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0)
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), 'surface')
    reader = UsdShade.Shader.Define(stage, '/Canyon/Rock/UV')
    reader.CreateIdAttr('UsdPrimvarReader_float2')
    reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
    repeat = UsdShade.Shader.Define(stage, '/Canyon/Rock/Repeat')
    repeat.CreateIdAttr('UsdTransform2d')
    repeat.CreateInput('in', Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), 'result')
    repeat.CreateInput('scale', Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(2000, 2000))
    for label, file, input_name, detail in (
        ('Albedo', 'albedo.jpg', 'diffuseColor', False),
        ('Normal', 'rock_normal.png', 'normal', True),
        ('Roughness', 'roughness.png', 'roughness', True),
    ):
        texture = UsdShade.Shader.Define(stage, f'/Canyon/Rock/{label}')
        texture.CreateIdAttr('UsdUVTexture')
        texture.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('./' + file))
        texture.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('raw' if detail else 'sRGB')
        texture.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource((repeat if detail else reader).ConnectableAPI(), 'result')
        texture.CreateInput('wrapS', Sdf.ValueTypeNames.Token).Set('repeat')
        texture.CreateInput('wrapT', Sdf.ValueTypeNames.Token).Set('repeat')
        if label == 'Normal':
            texture.CreateInput('scale', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
            texture.CreateInput('bias', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
        surface.CreateInput(input_name, Sdf.ValueTypeNames.Float if label == 'Roughness' else Sdf.ValueTypeNames.Float3).ConnectToSource(texture.ConnectableAPI(), 'r' if label == 'Roughness' else 'rgb')
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()
    # Select only complete triangles in the roughly 47m square around the robot.
    selected = faces[np.all(np.max(np.abs(points[faces, :2]), axis=2) <= 24, axis=1)]
    unique, remapped = np.unique(selected, return_inverse=True)
    collision = stage_at('local_ground.usdc')
    ground = UsdGeom.Mesh.Define(collision, '/Canyon/Ground')
    ground.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points[unique]))
    ground.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(selected), 3, dtype=np.int32)))
    ground.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(remapped.astype(np.int32)))
    ground.CreateSubdivisionSchemeAttr('none')
    ground.CreateVisibilityAttr('invisible')
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(ground.GetPrim()).CreateApproximationAttr('none')
    collision.GetRootLayer().Save()
    report = {'visual_vertices': len(points), 'collision_vertices': len(unique), 'collision_triangles': len(selected),
              'collision_width_m': float(np.ptp(points[unique, 0])), 'albedo_resolution': list(image.size),
              'detail_resolution': [512, 512], 'spawn_z': 0, 'seed': 7}
    (output / 'build.json').write_text(json.dumps(report, indent=2) + '\n')
    print(report)


if __name__ == '__main__':
    build()
