"""Experimental two-way Newton MPM soil for the single-environment canyon preview.

Material values are provisional. Loading/coupling is not material validation.
The substrate is lowered so the 16 cm soil layer replaces the original terrain.
"""

import json
import math

import numpy as np

from mars_project.environment.regolith import terrain_height
from mars_project.scenes.canyon import OPTIMIZED_DIR

X_MIN, X_MAX = -1.5, 5.0
Y_MIN, Y_MAX = -2.0, 2.0
THICKNESS = 0.16
VOXEL_SIZE = 0.04
DENSITY = 1400.0
ROBOT_PATTERN = r"/World/envs/env_.*/Robot"
BED_PATTERN = r"/World/envs/env_.*/SoilBed"
# PhysX ordering used by the existing rough-H1 checkpoint and recorded baseline.
# Newton imports the articulation depth-first, so policy I/O must select by name.
POLICY_JOINT_NAMES = (
    "left_hip_yaw",
    "right_hip_yaw",
    "torso",
    "left_hip_roll",
    "right_hip_roll",
    "left_shoulder_pitch",
    "right_shoulder_pitch",
    "left_hip_pitch",
    "right_hip_pitch",
    "left_shoulder_roll",
    "right_shoulder_roll",
    "left_knee",
    "right_knee",
    "left_shoulder_yaw",
    "right_shoulder_yaw",
    "left_ankle",
    "right_ankle",
    "left_elbow",
    "right_elbow",
)


def soil_particles(voxel_size=VOXEL_SIZE):
    """Equal-volume cell centers following the packaged canyon substrate."""
    if not math.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("MPM voxel size must be finite and positive")
    lower = np.array([X_MIN, Y_MIN, 0.0])
    upper = np.array([X_MAX, Y_MAX, THICKNESS])
    counts = np.ceil((upper - lower) / (voxel_size / 2)).astype(int)
    spacing = (upper - lower) / counts
    points = np.stack(
        np.meshgrid(*[lower[i] + (np.arange(counts[i]) + 0.5) * spacing[i] for i in range(3)], indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    points += np.random.default_rng(42).uniform(-0.05, 0.05, points.shape) * spacing
    points[:, 2] += terrain_height(points[:, 0], points[:, 1]) - THICKNESS
    volume = float(np.prod(spacing))
    return points.astype(np.float32), DENSITY * volume, float(np.cbrt(volume) / 2)


def spawn_soil_bed(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Import the existing invisible mesh as a kinematic body shared by both solvers."""
    from pxr import UsdPhysics

    from isaaclab.sim.spawners.from_files import spawn_from_usd

    prim = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    body = UsdPhysics.RigidBodyAPI.Apply(prim)
    body.CreateRigidBodyEnabledAttr(True)
    body.CreateKinematicEnabledAttr(True)
    return prim


def create_soil_height_scanner(cfg):
    """Resolve the mesh adapter after Kit has selected its USD libraries."""
    from isaaclab_newton.sensors.ray_caster.legacy_ray_caster import LegacyRayCaster

    return LegacyRayCaster(cfg)


def initialize_local_particles(_):
    """Exclude distant material before the solver allocates its first sparse grid."""
    from isaaclab.sim import SimulationContext
    from isaaclab_newton.physics.newton_manager_cfg import NewtonBuilderCfg

    sim = SimulationContext.instance()
    builder = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=sim.cfg.physics))
    points = np.asarray(builder.particle_q)
    builder.particle_flags = ((points[:, 0] ** 2 + points[:, 1] ** 2) < 1.0).astype(np.int32).tolist()


def spawn_local_soil(prim_path, cfg, translation=None, orientation=None, **kwargs):
    from isaaclab.physics import PhysicsEvent
    from isaaclab_newton.physics import NewtonManager
    from isaaclab_newton.sim.spawners.mpm.mpm import spawn_mpm_particles

    NewtonManager.register_callback(initialize_local_particles, PhysicsEvent.MODEL_INIT, name="local_soil_mask")
    return spawn_mpm_particles(prim_path, cfg, translation, orientation, **kwargs)


def configure_mpm_soil(cfg):
    from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

    import isaaclab.sim as sim
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
    from isaaclab_newton.assets import MPMObjectCfg
    from isaaclab_newton.physics import MJWarpSolverCfg, MPMSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg
    from isaaclab_newton.sim.schemas import NewtonCollisionCfg
    from isaaclab_newton.sim.spawners.mpm import MPMParticleMaterialCfg, MPMPointsCfg

    if cfg.scene.num_envs != 1:
        raise ValueError("The canyon MPM preview supports one environment")
    points, mass, radius = soil_particles()
    cfg.scene.terrain = None
    cfg.scene.env_spacing = 0.0
    cfg.sim.render_interval = cfg.decimation * 2
    cfg.scene.soil_bed = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SoilBed",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -THICKNESS)),
        spawn=sim.UsdFileCfg(
            func=spawn_soil_bed,
            usd_path=str(OPTIMIZED_DIR / "local_ground.usdc"),
            collision_props=[
                sim.UsdPhysicsCollisionCfg(collision_enabled=True),
                NewtonCollisionCfg(contact_margin=0.0, contact_gap=0.0),
            ],
        ),
    )
    cfg.scene.soil = MPMObjectCfg(
        prim_path="{ENV_REGEX_NS}/Soil",
        spawn=MPMPointsCfg(
            func=spawn_local_soil,
            positions=points,
            mass=mass,
            radius=radius,
            visible=False,
            material=MPMParticleMaterialCfg(density=DENSITY, young_modulus=1e7, friction=0.68),
            visual_color=(0.47, 0.20, 0.09),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(0.47, 0.20, 0.09), roughness=0.95),
        ),
    )
    cfg.scene.robot.init_state.pos = (0.0, 0.0, 1.05)
    from mars_project.environment.soil_runtime import reset_soil, update_soil

    cfg.events.soil_reset = EventTermCfg(func=reset_soil, mode="reset")
    cfg.events.soil_update = EventTermCfg(func=update_soil, mode="interval", interval_range_s=(0.0, 0.0))
    cfg.scene.robot.spawn.make_uninstanceable = True
    cfg.actions.joint_pos.joint_names = list(POLICY_JOINT_NAMES)
    cfg.actions.joint_pos.preserve_order = True
    for name in ("joint_pos", "joint_vel"):
        getattr(cfg.observations.policy, name).params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=list(POLICY_JOINT_NAMES), preserve_order=True
        )
    cfg.scene.robot.spawn.collision_props = [
        sim.UsdPhysicsCollisionCfg(collision_enabled=True),
        NewtonCollisionCfg(contact_margin=0.0, contact_gap=0.0),
    ]
    cfg.scene.height_scanner.mesh_prim_paths = ["/World/envs/env_0/SoilBed"]
    # The Newton rough-task preset scans only the global world, while this bed
    # belongs to env_0. Use its explicit-mesh adapter to see the bed and avoid
    # ray hits on the robot itself; the observation then adds the live MPM tops.
    cfg.scene.height_scanner.class_type = create_soil_height_scanner
    cfg.observations.policy.height_scan.func = mpm_height_scan
    # Coupled Newton does not expose the per-entry forces through ContactSensor.
    cfg.scene.contact_forces = None
    cfg.scene.robot.spawn.activate_contact_sensors = False
    cfg.events.physics_material = None
    cfg.rewards.feet_air_time = None
    cfg.rewards.feet_slide = None
    cfg.rewards.undesired_contacts = None
    cfg.terminations.base_contact = None
    cfg.terminations.soil_fall = TerminationTermCfg(func=soil_fall)
    cfg.sim.physics = NewtonCfg(
        use_cuda_graph=True,
        solver_cfg=CouplerProxyCfg(
            entries=[
                CouplerEntryCfg(
                    name="rigid",
                    solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=False, njmax=1000, nconmax=300),
                    bodies=[ROBOT_PATTERN, BED_PATTERN],
                    substeps=2,
                ),
                CouplerEntryCfg(
                    name="mpm",
                    solver_cfg=MPMSolverCfg(
                        voxel_size=VOXEL_SIZE,
                        grid_type="sparse",
                        grid_padding=0,
                        # Active-region capacity with headroom for disturbed
                        # material, leaving VRAM for Kit's renderer.
                        max_active_cell_count=1 << 16,
                        max_iterations=100,
                        tolerance=1e-5,
                        strain_basis="P0",
                        air_drag=1.0,
                    ),
                    all_particles=True,
                    include_child_joints=False,
                    in_place=True,
                ),
            ],
            proxies=[
                CouplerProxyMappingCfg(
                    source="rigid",
                    destination="mpm",
                    bodies=[ROBOT_PATTERN, BED_PATTERN],
                    mode="lagged",
                    mass_scale=1.0,
                    proxy_relaxation=1.0,
                    collision_pipeline=None,
                )
            ],
            iterations=1,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(soft_contact_max=0),
        load_visual_shapes=False,
    )


def particle_surface_heights(points, query_xy, fallback, radius, cell=VOXEL_SIZE):
    """Bin live material-point tops on the GPU; uncovered queries use the substrate.

    This is a height-field observation of MPM state, not a collision surface or
    a prescribed deformation. It includes material that moves outside the initial bed.
    """
    import torch

    # The ray-scanned corridor bounds enclose the soil and its free edges.
    xmin, ymin, xmax, ymax = -1.5, -2.0, 5.0, 2.0
    nx, ny = math.ceil((xmax - xmin) / cell), math.ceil((ymax - ymin) / cell)
    ix = torch.floor((points[:, 0] - xmin) / cell).long()
    iy = torch.floor((points[:, 1] - ymin) / cell).long()
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    tops = torch.full((nx * ny,), -torch.inf, device=points.device, dtype=points.dtype)
    tops.scatter_reduce_(0, iy[valid] * nx + ix[valid], points[valid, 2] + radius, reduce="amax")
    qx = torch.floor((query_xy[..., 0] - xmin) / cell).long()
    qy = torch.floor((query_xy[..., 1] - ymin) / cell).long()
    inside = (qx >= 0) & (qx < nx) & (qy >= 0) & (qy < ny)
    sampled = tops[qy.clamp(0, ny - 1) * nx + qx.clamp(0, nx - 1)]
    return torch.where(inside, torch.maximum(sampled, fallback), fallback)


def mpm_height_scan(env, sensor_cfg, offset=0.5):
    sensor = env.scene[sensor_cfg.name]
    hits = sensor.data.ray_hits_w.torch
    if not hits.isfinite().all():
        raise RuntimeError(
            f"MPM height scan missed substrate: root={env.scene['robot'].data.root_pos_w.torch.tolist()} "
            f"sensor={sensor.data.pos_w.torch.tolist()} misses={int((~hits.isfinite().all(-1)).sum())}"
        )
    soil = env.scene["soil"]
    if hasattr(env, "soil_runtime"):
        runtime = env.soil_runtime
        heights = hits[..., 2] + THICKNESS + runtime.displacement_at(hits[0, :, :2])[None]
    else:
        heights = particle_surface_heights(
            soil.data.particle_pos_w.torch[0], hits[..., :2], hits[..., 2], soil.cfg.spawn.radius
        )
    return sensor.data.pos_w.torch[:, 2, None] - heights - offset


def soil_fall(env):
    """Keep fall detection when contact-force sensors are unavailable."""
    robot = env.scene["robot"].data
    hits = env.scene["height_scanner"].data.ray_hits_w.torch
    ground = hits[..., 2].median(dim=1).values
    return (robot.root_pos_w.torch[:, 2] - ground < 0.55) | (robot.projected_gravity_b.torch[:, 2] > -0.5)


def verify_soil_loaded(env):
    """Check actual solver state and USD visibility; fail instead of silently using rigid ground."""
    from pxr import UsdGeom

    from isaaclab.sim.utils.stage import get_current_stage
    from isaaclab_newton.physics import NewtonManager

    soil = env.scene["soil"]
    model = NewtonManager.get_model()
    mpm = NewtonManager._solver.solver("mpm")
    if not np.isclose(float(env.scene["soil_bed"].data.root_pos_w.torch[0, 2]), -THICKNESS, atol=1e-6):
        raise RuntimeError("The collision substrate was not lowered beneath the original terrain surface")
    points = soil.data.particle_pos_w.torch
    if soil.particles_per_object != len(soil.cfg.spawn.positions) or not points.isfinite().all():
        raise RuntimeError("MPM soil did not load finite material points")
    expected_mass = (X_MAX - X_MIN) * (Y_MAX - Y_MIN) * THICKNESS * DENSITY
    loaded_mass = (
        float(env.soil_runtime.mass.double().sum())
        if hasattr(env, "soil_runtime")
        else float(model.particle_mass.numpy().sum())
    )
    if not np.isclose(loaded_mass, expected_mass, rtol=1e-5):
        raise RuntimeError("Loaded MPM soil mass does not match the configured layer")
    density = mpm._mpm_model.particle_density.numpy()
    if not np.all(np.isclose(density, DENSITY, rtol=1e-5) | (density == 0)):
        raise RuntimeError("Loaded MPM material-point density is incorrect")
    stage = get_current_stage()
    cloud = stage.GetPrimAtPath("/World/envs/env_0/Soil/Particles")
    if cloud and UsdGeom.Imageable(cloud).ComputeVisibility() != "invisible":
        raise RuntimeError("Solver material points must not render as oversized grains")
    print(
        "MPM_LOADED "
        + json.dumps(
            dict(
                particles=soil.particles_per_object,
                mass_kg=loaded_mass,
                density_kg_m3=DENSITY,
                voxel_size_m=VOXEL_SIZE,
                thickness_m=THICKNESS,
                coupling="two_way",
                visualization="continuous_textured_terrain",
                material_validated=False,
            )
        ),
        flush=True,
    )


def soil_progress(env):
    """Report solved robot/soil interaction and reject nonfinite runtime state."""
    from isaaclab_newton.physics import NewtonManager

    soil = env.scene["soil"]
    points = soil.data.particle_pos_w.torch
    if not points.isfinite().all() or not soil.data.particle_vel_w.torch.isfinite().all():
        raise RuntimeError("MPM soil became nonfinite")
    coupled = NewtonManager._solver
    mpm = coupled.solver("mpm")
    mpm.check_sparse_grid_rebuild_status()
    impulses, _, ids = mpm.collect_collider_impulses(coupled.entry_state("mpm"))
    collider_ids = ids.numpy()
    body_ids = mpm.collider_body_index.numpy()
    valid = (collider_ids >= 0) & (collider_ids < len(body_ids))
    robot_body_ids = {i for i, label in enumerate(coupled.view("mpm").body_label) if "/Robot/" in label}
    on_robot = np.zeros(len(collider_ids), dtype=bool)
    on_robot[valid] = np.isin(body_ids[collider_ids[valid]], list(robot_body_ids))
    force = impulses.numpy()[on_robot].sum(axis=0) / env.physics_dt
    displacement = (points - soil.data.default_particle_state_w.torch[..., :3]).square().sum(-1).mean().sqrt()
    print(
        f"MPM_PROGRESS particles={soil.particles_per_object} rms_displacement_m={float(displacement):.6f} "
        f"robot_soil_force_n={force.tolist()} finite=True",
        flush=True,
    )
    if hasattr(env, "soil_runtime"):
        runtime = env.soil_runtime
        cloud = runtime.dust
        ledger = float(runtime.mass.double().sum()) + cloud.mass.sum() + cloud.deposited
        print(
            f"LOCAL_SOIL dynamic={int(runtime.dynamic.sum())} halo={int((runtime.active & ~runtime.dynamic).sum())} "
            f"sleeping={int((~runtime.active).sum())} dust_parcels={len(cloud.mass)} "
            f"dust_emitted_kg={cloud.emitted:.9f} dust_airborne_kg={cloud.mass.sum():.9f} "
            f"dust_deposited_kg={cloud.deposited:.9f} total_mass_kg={ledger:.6f}",
            flush=True,
        )
        print(
            f"DUST_MODEL max_resolved_dust_gas_mass_ratio={runtime.dust_loading:.6f} "
            f"peak_dust_gas_mass_ratio={runtime.peak_dust_loading:.6f} validated_material=False",
            flush=True,
        )
    return force
