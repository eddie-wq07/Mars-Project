"""Configure the standalone, visual-only Verdant Trail preview before physics starts."""

import json

from mars_project.assets.scene_paths import EARTH_DIR

USD_DIR = EARTH_DIR / "usd"
EARTH_ASSET = USD_DIR / "verdant_trail_earth.usdc"

# Keep source HDR radiance; the illuminating map already contains a high-dynamic-range sun.
EARTH_LIGHT_EXPOSURE = 0.0
# Scene-referred HDRs at Blender exposure 0. RTX uses ISO=0 to bypass its
# physical-camera multiplier (same neutral preset as installed isaaclab_ppisp's
# rtx_camera_overrides.py). This is RTX-specific, not a physical ISO or the
# UsdGeomCamera.ComputeLinearExposureScale formula. ACES remains enabled.
# Exposure controls do not change motion blur or depth of field.
EARTH_CAMERA_EXPOSURE = {
    "exposure": 0.0,
    "exposure:iso": 0.0,
    "exposure:time": 1.0,
    "exposure:fStop": 1.0,
    "exposure:responsivity": 1.0,
}
EARTH_RENDER_SETTINGS = {
    "/rtx/rendermode": "RealTimePathTracing",
    "/rtx/pathtracing/spp": 2,
    "/rtx/pathtracing/totalSpp": 64,
    "/rtx/pathtracing/optixDenoiser/enabled": True,
    "/rtx/post/tonemap/op": 6,  # ACES in the installed RTX settings schema
    "/rtx/post/tonemap/filmIso": EARTH_CAMERA_EXPOSURE["exposure:iso"],
    "/rtx/post/tonemap/exposureTime": EARTH_CAMERA_EXPOSURE["exposure:time"],
    "/rtx/post/tonemap/fNumber": EARTH_CAMERA_EXPOSURE["exposure:fStop"],
    "/rtx/post/tonemap/responsivity": EARTH_CAMERA_EXPOSURE["exposure:responsivity"],
    "/rtx/post/tonemap/colorMode": 0,  # linear sRGB inputs
    "/rtx/post/tonemap/whitepoint": (1.0, 1.0, 1.0),
    "/rtx/raytracing/showLights": 0,  # respect visibleInPrimaryRay on each dome
    "/rtx/background/source/type": "domeLight",
    "/rtx/domeLight/upperLowerStrategy": 0,  # sample HDR sun, not a low-frequency approximation
    "/rtx/post/histogram/enabled": False,
    "/rtx/post/histogram/filterType": 0,  # median
    "/rtx/post/histogram/whiteScale": 1.0,
    "/rtx/post/histogram/tau": 1.0,
    "/rtx/post/histogram/useExposureClamping": False,
    "/rtx/post/tonemap/ocio/enabled": False,
    "/rtx/raytracing/globalVolumetricEffects/enabled": True,
    # RTX's legacy fog distance settings are centimeters, even on a meter stage.
    # T(d) = 0.9 ** (0.15 * d_cm / 1000): 0.984 at 10 m, 0.910 at 60 m.
    "/rtx/raytracing/inscattering/atmosphereHeight": 3000.0,
    "/rtx/raytracing/inscattering/maxDistance": 6000.0,
    "/rtx/raytracing/inscattering/densityMult": 0.15,
    "/rtx/raytracing/inscattering/transmittanceMeasurementDistance": 1000.0,
    "/rtx/raytracing/inscattering/transmittanceColor": (0.9, 0.9, 0.9),
    "/rtx/raytracing/inscattering/singleScatteringAlbedo": (0.95, 0.92, 0.86),
    "/rtx/raytracing/inscattering/anisotropyFactor": 0.6,
}


def author_earth_view_settings(stage, camera_path, render_product_path, render_mode="Real-Time Path-Tracing"):
    """Author effective per-view settings; also testable on an in-memory USD stage."""
    from pxr import Sdf, UsdGeom, UsdRender

    camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
    product = UsdRender.Product(stage.GetPrimAtPath(render_product_path))
    if not camera or not product:
        raise RuntimeError("Earth requires an existing viewport camera and render product")
    # Kit's USD watcher requires these APIs to route per-view settings to RTX.
    # AddAppliedSchema also works in offline USD without loading Kit plugins.
    for schema in ["OmniRtxCameraExposureAPI_1", "OmniRtxCameraAutoExposureAPI_1"]:
        camera.GetPrim().AddAppliedSchema(schema)
    for schema in ["OmniRtxSettingsCommonAPI_1", "OmniRtxPostTonemapAPI_1"]:
        product.GetPrim().AddAppliedSchema(schema)
    for name, value in EARTH_CAMERA_EXPOSURE.items():
        camera.GetPrim().CreateAttribute(name, Sdf.ValueTypeNames.Float, custom=False).Set(value)
    camera.GetPrim().CreateAttribute("omni:rtx:autoExposure:enabled", Sdf.ValueTypeNames.Bool, custom=False).Set(False)
    for name, value in {
        "omni:rtx:background:source:type": "domeLight",
        "omni:rtx:lights:visibilityOverride": "perLight",
        "omni:rtx:rendermode": render_mode,
        "omni:rtx:post:tonemap:operator": "acesApproximation",
        "omni:rtx:post:tonemap:colorMode": "sRGBLinear",
    }.items():
        product.GetPrim().CreateAttribute(name, Sdf.ValueTypeNames.Token).Set(value)


def apply_earth_render_settings():
    """Apply per-run exposure after environment creation; no persistent settings changed."""
    import carb.settings

    settings = carb.settings.get_settings()
    for name, value in EARTH_RENDER_SETTINGS.items():
        settings.set(name, value)
    from omni.kit.viewport.utility import get_active_viewport

    from isaaclab.sim.utils.stage import get_current_stage

    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("Earth appearance requires a Kit viewport (--viz kit)")
    author_earth_view_settings(get_current_stage(), viewport.camera_path, viewport.render_product_path)
    print(
        f"Earth: source HDRI, ACES, fixed camera exposure {EARTH_CAMERA_EXPOSURE}; "
        "RTX Real-Time 2.0 preview and light atmospheric scattering.",
        flush=True,
    )


def configure_earth_scene(cfg):
    """Register scenery with Isaac Lab's normal scene-construction lifecycle."""
    from isaaclab_visualizers.kit import KitVisualizerCfg

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    if not EARTH_ASSET.is_file():
        raise FileNotFoundError(f"Package Verdant Trail first: {EARTH_ASSET}")
    anchor = json.loads((USD_DIR / "conversion.json").read_text())["anchor"]
    cfg.scene.earth_scenery = AssetBaseCfg(
        prim_path="/World/VerdantTrail",
        spawn=sim_utils.UsdFileCfg(usd_path=str(EARTH_ASSET)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(-v for v in anchor)),
        cloning_contexts=(),
    )
    collision_asset = USD_DIR / "earth_collision_ground.usda"
    if not collision_asset.is_file():
        raise FileNotFoundError(collision_asset)
    cfg.scene.terrain.terrain_type = "usd"
    cfg.scene.terrain.usd_path = str(collision_asset)
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.terrain.visual_material = None

    look = json.loads((USD_DIR.parent / "reference-frame.json").read_text())
    camera = look["camera"]
    forward = tuple(-camera["matrix"][axis][2] for axis in range(3))
    source_eye = tuple(camera["position"][axis] - anchor[axis] for axis in range(3))
    # Match the source camera near 19 seconds (frame 457 at 24 fps).
    eye = source_eye
    lookat = tuple(source_eye[axis] + 6.0 * forward[axis] for axis in range(3))
    cfg.sim.default_visualizer_cfg = KitVisualizerCfg(
        eye=eye,
        lookat=lookat,
        focal_length=camera["lens_mm"],
        background_color=None,  # show the HDRI rather than the default flat blue background
    )
    # The packaged scene separates illumination from the camera-visible sky.
    cfg.scene.sky_light = None


def add_earth_scene():
    """Finish visual setup after creation; do not import geometry into a live physics scene."""
    from pxr import UsdGeom

    from isaaclab.sim.utils.stage import get_current_stage

    stage = get_current_stage()
    if not stage.GetPrimAtPath("/World/VerdantTrail").IsValid():
        raise RuntimeError("Earth scenery was not registered before environment creation")
    # Ground visibility is authored in the collision asset before physics/Fabric starts.
    ground = stage.GetPrimAtPath("/World/ground/terrain/Plane")
    if not ground or UsdGeom.Imageable(ground).ComputeVisibility() != "invisible":
        raise RuntimeError("Earth collision ground must be invisible at scene creation")
    print("Verdant Trail ready: textured trail, invisible collision ground, source-inspired camera/light.", flush=True)
