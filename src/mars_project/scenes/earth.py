"""Configure the standalone, visual-only Verdant Trail preview before physics starts."""

import json
import math

from mars_project.assets.scene_paths import EARTH_DIR

USD_DIR = EARTH_DIR / "usd"
EARTH_ASSET = USD_DIR / "verdant_trail_earth.usdc"

# Keep source HDR radiance; the illuminating map already contains a high-dynamic-range sun.
EARTH_LIGHT_EXPOSURE = 0.0
EARTH_RENDER_SETTINGS = {
    "/rtx/rendermode": "RealTimePathTracing",
    "/rtx/pathtracing/spp": 2,
    "/rtx/pathtracing/totalSpp": 64,
    "/rtx/pathtracing/optixDenoiser/enabled": True,
    "/rtx/raytracing/showLights": 0,  # respect visibleInPrimaryRay on each dome
    "/rtx/background/source/type": "domeLight",
    "/rtx/domeLight/upperLowerStrategy": 0,  # sample HDR sun, not a low-frequency approximation
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


def prepare_earth_renderer():
    """Configure asset loading and HDR sampling before the scene reaches RTX."""
    import carb.settings

    settings = carb.settings.get_settings()
    for name in ("/rtx/materialDb/syncLoads", "/rtx/hydra/materialSyncLoads", "/omni/kit/plugin/syncUsdLoads"):
        settings.set(name, True)
    for name, value in EARTH_RENDER_SETTINGS.items():
        settings.set(name, value)


def apply_earth_render_settings():
    """Apply per-run exposure after environment creation; no persistent settings changed."""
    prepare_earth_renderer()
    from omni.kit.viewport.utility import get_active_viewport

    from isaaclab.sim.utils.stage import get_current_stage

    from mars_project.scenes.cameras import apply_view_settings

    # Initialize the existing view BEFORE env.reset() renders and the runner switches cameras.
    # Preserve the earlier Earth initialization sequence as well as the final view settings.
    # Keep the later application too, so the final camera has explicit exposure settings.
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("Earth appearance requires a Kit viewport")
    apply_view_settings(get_current_stage(), viewport.camera_path, viewport.render_product_path)
    print("Earth: source HDRI and light atmospheric scattering; initial view settings applied.", flush=True)


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
    # The source camera sits on the outward trail, looking into the cave.
    # Face from the fixed spawn toward that landmark, not an arbitrary world yaw.
    cfg.seed = 42
    cfg.scene.robot.init_state.pos = (0.0, 0.0, 1.05)
    cfg.scene.robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)  # Isaac Lab uses xyzw
    outward_yaw = math.atan2(source_eye[1], source_eye[0])
    cfg.events.reset_base.params["pose_range"] = {"yaw": (outward_yaw, outward_yaw)}
    cfg.events.reset_base.params["velocity_range"] = {}
    cfg.scene.robot.init_state.lin_vel = (0.0, 0.0, 0.0)
    cfg.scene.robot.init_state.ang_vel = (0.0, 0.0, 0.0)
    cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    cfg.events.add_base_mass = None
    cfg.events.base_external_force_torque = None
    print(f"EARTH_SPAWN position={cfg.scene.robot.init_state.pos} outward_yaw_deg={math.degrees(outward_yaw):.3f}")
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
