"""Canyon appearance and bounded collision terrain; no launcher logic."""

from mars_project.assets.scene_paths import GRAND_CANYON_DIR

OPTIMIZED_DIR = GRAND_CANYON_DIR / "optimized"


def configure_canyon_scene(cfg):
    """Spawn scenery and the local collision mesh before physics initializes."""
    from isaaclab_visualizers.kit import KitVisualizerCfg

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    for name in ("scenery.usdc", "local_ground.usdc", "sky.png"):
        if not (OPTIMIZED_DIR / name).is_file():
            raise FileNotFoundError(f"Restore packaged canyon assets with git lfs pull: ({name})")
    cfg.scene.canyon_scenery = AssetBaseCfg(
        prim_path="/World/MarsLabScenery",
        spawn=sim_utils.UsdFileCfg(usd_path=str(OPTIMIZED_DIR / "scenery.usdc")),
        cloning_contexts=(),
    )
    cfg.scene.terrain.terrain_type = "usd"
    cfg.scene.terrain.usd_path = str(OPTIMIZED_DIR / "local_ground.usdc")
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.terrain.visual_material = None
    cfg.scene.terrain.debug_vis = False
    # Neutral sky fill preserves rock colors; a warm sun supplies directional shape.
    cfg.scene.sky_light.spawn = sim_utils.DomeLightCfg(
        intensity=2.0,
        color=(1.0, 1.0, 1.0),
        visible_in_primary_ray=True,
        texture_file=str(OPTIMIZED_DIR / "sky.png"),
        texture_format="latlong",
    )
    cfg.scene.sky_light.init_state.rot = (0.7071068, 0.0, 0.0, 0.7071068)
    cfg.scene.canyon_sun = AssetBaseCfg(
        prim_path="/World/CanyonSun",
        spawn=sim_utils.DistantLightCfg(intensity=3.0, color=(1.0, 0.88, 0.72), angle=0.7),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.2588190, 0.0, 0.0, 0.9659258)),
        cloning_contexts=(),
    )
    cfg.sim.default_visualizer_cfg = KitVisualizerCfg(
        eye=(4.0, -6.0, 3.0), lookat=(0.0, 0.0, 0.9), background_color=None
    )
    # Keep random reset positions on the gentle central part of the local surface.
    cfg.events.reset_base.params["pose_range"].update(x=(-0.3, 0.3), y=(-0.3, 0.3))


def add_canyon_scene():
    """Apply lightweight render settings and verify the prebuilt terrain."""
    import carb.settings
    from pxr import Sdf, UsdGeom, UsdLux, UsdPhysics

    from isaaclab.sim.utils.stage import get_current_stage

    stage = get_current_stage()
    for prim in stage.Traverse():
        if prim.IsA(UsdLux.DomeLight):
            prim.CreateAttribute("omni:rtx:usdluxVersion", Sdf.ValueTypeNames.Int).Set(2505)
    ground = stage.GetPrimAtPath("/World/ground/terrain/Ground")
    if not ground or not ground.HasAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError("Local canyon collision mesh was not loaded")
    if UsdGeom.Imageable(ground).ComputeVisibility() != "invisible":
        raise RuntimeError("Collision-only canyon mesh must be invisible")
    settings = carb.settings.get_settings()
    for key, value in {
        "/rtx/rendermode": "RealTimePathTracing",
        "/rtx/post/tonemap/op": 6,
        "/rtx/post/histogram/enabled": False,
        "/rtx/post/tonemap/filmIso": 0.0,
        "/rtx/post/tonemap/exposureTime": 1.0,
        "/rtx/post/tonemap/fNumber": 1.0,
        "/rtx/post/tonemap/responsivity": 1.0,
        "/rtx/background/source/type": "domeLight",
        "/rtx/raytracing/showLights": 0,
        "/rtx/raytracing/globalVolumetricEffects/enabled": False,
        "/app/viewport/grid/enabled": False,
    }.items():
        settings.set(key, value)
    print(
        "Canyon: 4K albedo, tiled rock normals, sun/sky lighting; local uneven collision mesh, no flat plane.",
        flush=True,
    )
