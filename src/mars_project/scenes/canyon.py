"""Canyon appearance and bounded collision terrain; no launcher logic."""

from mars_project.assets.scene_paths import GRAND_CANYON_DIR

OPTIMIZED_DIR = GRAND_CANYON_DIR / "optimized"

# Reduce sky illumination without dimming the camera-visible orange background.
SKY_SURFACE_CONTRIBUTION = 0.25


def prepare_soil_renderer():
    """Select heterogeneous-volume rendering before scenery/camera activation."""
    import carb.settings

    settings = carb.settings.get_settings()
    for key, value in {
        "/rtx/rendermode": "PathTracing",
        "/rtx/pathtracing/spp": 4,
        "/rtx/pathtracing/totalSpp": 128,
        "/rtx/pathtracing/optixDenoiser/enabled": True,
        "/rtx/pathtracing/ptvol/enabled": True,
    }.items():
        settings.set(key, value)


def configure_canyon_scene(cfg, soil="mpm"):
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
    # Orange sky background; surface contribution is reduced after scene creation.
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
    # Short, repeatable +X route inside the local collision corridor.
    cfg.seed = 42
    cfg.scene.robot.init_state.pos = (0.0, 0.0, 1.05)
    cfg.scene.robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)
    cfg.scene.robot.init_state.lin_vel = (0.0, 0.0, 0.0)
    cfg.scene.robot.init_state.ang_vel = (0.0, 0.0, 0.0)
    cfg.events.reset_base.params["pose_range"] = {"yaw": (0.0, 0.0)}
    cfg.events.reset_base.params["velocity_range"] = {}
    cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    cfg.events.add_base_mass = None
    cfg.events.base_external_force_torque = None
    if soil == "mpm":
        from mars_project.environment.mpm_soil import configure_mpm_soil

        configure_mpm_soil(cfg)
    elif soil != "rigid":
        raise ValueError(f"Unknown soil model: {soil}")


def add_canyon_scene():
    """Apply lightweight render settings and verify the prebuilt terrain."""
    import carb.settings
    from pxr import Sdf, UsdGeom, UsdLux, UsdPhysics

    from isaaclab.sim.utils.stage import get_current_stage

    stage = get_current_stage()
    for prim in stage.Traverse():
        if prim.IsA(UsdLux.DomeLight):
            prim.CreateAttribute("omni:rtx:usdluxVersion", Sdf.ValueTypeNames.Int).Set(2505)
            light = UsdLux.LightAPI(prim)
            light.CreateDiffuseAttr(SKY_SURFACE_CONTRIBUTION)
            light.CreateSpecularAttr(SKY_SURFACE_CONTRIBUTION)
    ground = stage.GetPrimAtPath("/World/envs/env_0/SoilBed/Ground")
    if not ground:
        ground = stage.GetPrimAtPath("/World/ground/terrain/Ground")
    if not ground or not ground.HasAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError("Local canyon collision mesh was not loaded")
    if UsdGeom.Imageable(ground).ComputeVisibility() != "invisible":
        raise RuntimeError("Collision-only canyon mesh must be invisible")
    settings = carb.settings.get_settings()
    for key, value in {
        "/rtx/rendermode": "PathTracing" if stage.GetPrimAtPath("/World/envs/env_0/SoilBed") else "RealTimePathTracing",
        "/rtx/background/source/type": "domeLight",
        "/rtx/raytracing/showLights": 0,
        "/rtx/raytracing/globalVolumetricEffects/enabled": False,
        "/app/viewport/grid/enabled": False,
    }.items():
        settings.set(key, value)
    print(
        "Canyon: 4K albedo, tiled rock normals, sun/sky lighting; local uneven substrate loaded.",
        flush=True,
    )
