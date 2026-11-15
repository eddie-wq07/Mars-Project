"""Run paired H1 trials with unchanged policy weights; save every 50 Hz trajectory.

Run each condition in a fresh process with --condition earth or mars, sharing
--output-dir and --seed. No renderer is used; preview/capture startup is untouched.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from isaaclab.app import add_launcher_args, launch_simulation

from mars_project.evaluation import Protocol, classify_sample, paired_start, route_rotation, summarize


def configure_evaluation(condition):
    from isaaclab.managers import SceneEntityCfg
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg
    from isaaclab_newton.sim.schemas import NewtonCollisionCfg

    from isaaclab_tasks.utils import parse_env_cfg

    import mars_project.tasks  # noqa: F401
    from mars_project.environment.mpm_soil import POLICY_JOINT_NAMES, create_soil_height_scanner
    from mars_project.scenes.setup import configure_preview

    cfg = parse_env_cfg("Mars-Project-H1", num_envs=1, overrides=["physics=newton_mjwarp"])
    cfg.sim.gravity = (0.0, 0.0, -9.81 if condition == "earth" else -3.71)
    configure_preview(
        cfg, "earth" if condition == "earth" else "grand-canyon", 0.5, soil="rigid" if condition == "earth" else "mpm"
    )
    route_yaw = cfg.events.reset_base.params["pose_range"]["yaw"][0]
    # Use the identical rigid solver and substeps as the accepted Mars MPM run.
    # Earth has no deformable solver to couple; use MJWarp directly with two substeps.
    if condition == "earth":
        import isaaclab.sim as sim

        cfg.sim.physics = NewtonCfg(
            use_cuda_graph=True,
            solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=False, njmax=1000, nconmax=300),
            num_substeps=2,
            collision_cfg=NewtonCollisionPipelineCfg(soft_contact_max=0),
            load_visual_shapes=False,
        )
        cfg.scene.robot.spawn.make_uninstanceable = True
        cfg.scene.robot.spawn.collision_props = [
            sim.UsdPhysicsCollisionCfg(collision_enabled=True),
            NewtonCollisionCfg(contact_margin=0.0, contact_gap=0.0),
        ]
        cfg.scene.height_scanner.class_type = create_soil_height_scanner
        cfg.scene.height_scanner.mesh_prim_paths = ["/World/ground"]
    # Remove irrelevant scenery for headless measurement, retaining collision and air geometry.
    for name in ("earth_scenery", "canyon_scenery", "sky_light", "canyon_sun"):
        if hasattr(cfg.scene, name):
            setattr(cfg.scene, name, None)
    cfg.actions.joint_pos.joint_names = list(POLICY_JOINT_NAMES)
    cfg.actions.joint_pos.preserve_order = True
    for name in ("joint_pos", "joint_vel"):
        getattr(cfg.observations.policy, name).params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=list(POLICY_JOINT_NAMES), preserve_order=True
        )
    cfg.scene.contact_forces = None
    cfg.scene.robot.spawn.activate_contact_sensors = False
    cfg.events.physics_material = None
    cfg.rewards = {}
    # Explicit scoring below prevents autoreset from erasing terminal state.
    cfg.terminations = {}
    cfg.episode_length_s = 30.0
    cfg.ui_window_class_type = None
    return cfg, route_yaw


def set_start(env, sample, yaw):
    rotation = route_rotation(yaw)
    xy = rotation @ [sample["forward_m"], sample["lateral_m"]]
    velocity = rotation @ [sample["forward_velocity_m_s"], sample["lateral_velocity_m_s"]]
    cfg = env.event_manager.get_term_cfg("reset_base")
    cfg.params["pose_range"] = {
        "x": (xy[0], xy[0]),
        "y": (xy[1], xy[1]),
        "yaw": (yaw + sample["yaw_rad"], yaw + sample["yaw_rad"]),
    }
    cfg.params["velocity_range"] = {
        "x": (velocity[0], velocity[0]),
        "y": (velocity[1], velocity[1]),
        "yaw": (sample["yaw_rate_rad_s"], sample["yaw_rate_rad_s"]),
    }
    env.event_manager.set_term_cfg("reset_base", cfg)


def verify_reset(env):
    import torch

    if hasattr(env, "soil_runtime"):
        soil, runtime = env.scene["soil"], env.soil_runtime
        assert torch.allclose(
            soil.data.particle_pos_w.torch, soil.data.default_particle_state_w.torch[..., :3], atol=1e-6
        )
        assert torch.count_nonzero(soil.data.particle_vel_w.torch) == 0
        assert torch.count_nonzero(runtime.displacement) == 0
        assert torch.equal(runtime.mass, runtime.base_mass)
        assert runtime.dust.emitted == 0 and len(runtime.dust.mass) == 0
    assert torch.count_nonzero(env.air_drag.force_w) == 0


def run_trial(env, wrapped, policy, condition, sample, yaw, output, protocol):
    import torch

    set_start(env, sample, yaw)
    env.reset(seed=sample["seed"])
    verify_reset(env)
    policy.reset(torch.ones(1, device=env.device, dtype=torch.bool))
    observations = wrapped.get_observations()
    robot = env.scene["robot"]
    foot_ids, _ = robot.find_bodies(".*_ankle_link")
    rotation = route_rotation(yaw)
    start = robot.data.root_pos_w.torch[0].cpu().numpy().copy()
    target = start[:2] + rotation @ [protocol.goal_m, 0.0]
    expected = rotation @ [sample["forward_m"], sample["lateral_m"]]
    assert np.allclose(start[:2], expected, atol=1e-5), (start, expected)
    initial_velocity = robot.data.root_vel_w.torch[0].cpu().numpy().copy()
    expected_velocity = rotation @ [sample["forward_velocity_m_s"], sample["lateral_velocity_m_s"]]
    assert np.allclose(initial_velocity[:2], expected_velocity, atol=1e-5)
    assert abs(initial_velocity[5] - sample["yaw_rate_rad_s"]) < 1e-5
    traces = {
        k: []
        for k in (
            "time_s",
            "root_position_w",
            "root_quaternion_xyzw",
            "root_velocity_w",
            "joint_position",
            "actions",
            "foot_position_w",
            "route_position",
            "root_height_m",
            "up_cosine",
            "air_drag_force_n",
            "min_soil_displacement_m",
            "max_soil_displacement_m",
        )
    }
    t0 = time.monotonic()
    max_mass_error = 0.0
    outcome = None
    actions = torch.zeros((1, 19), device=env.device)
    for step in range(round(protocol.timeout_s / env.step_dt) + 1):
        if step:
            assert all(torch.isfinite(v).all() for v in observations.values()), "Nonfinite policy input"
            actions = policy(observations)
            assert torch.isfinite(actions).all(), "Nonfinite policy action"
            observations, _, dones, _ = wrapped.step(actions)
            assert not dones.any(), "Unexpected autoreset invalidated trial"
        data = robot.data
        position = data.root_pos_w.torch[0].cpu().numpy().copy()
        if step:
            assert np.linalg.norm(position - traces["root_position_w"][-1]) < 0.25, "Discontinuous robot state"
        hits = env.scene["height_scanner"].data.ray_hits_w.torch[0]
        assert torch.isfinite(hits).all(), "Height scanner left the measured corridor"
        # Use nearby terrain under the base, rather than the sloped scan's median.
        nearest = ((hits[:, :2] - data.root_pos_w.torch[0, :2]) ** 2).sum(-1).argmin()
        ground = float(hits[nearest, 2])
        minimum = maximum = 0.0
        if hasattr(env, "soil_runtime"):
            from mars_project.environment.mpm_soil import THICKNESS

            runtime = env.soil_runtime
            ground += THICKNESS + float(runtime.displacement_at(data.root_pos_w.torch[:, :2])[0])
            minimum, maximum = float(runtime.displacement.min()), float(runtime.displacement.max())
            ledger = float(runtime.mass.double().sum()) + float(runtime.dust.mass.sum()) + runtime.dust.deposited
            error = abs(ledger - float(runtime.base_mass.double().sum()))
            max_mass_error = max(max_mass_error, error)
            assert error < 1e-5, f"Soil mass drift {error} kg"
            assert torch.isfinite(env.scene["soil"].data.particle_pos_w.torch).all()
        height = position[2] - ground
        up = -float(data.projected_gravity_b.torch[0, 2])
        route = (position[:2] - start[:2]) @ rotation
        values = dict(
            time_s=step * env.step_dt,
            root_position_w=position,
            root_quaternion_xyzw=data.root_quat_w.torch[0].cpu().numpy().copy(),
            root_velocity_w=data.root_vel_w.torch[0].cpu().numpy().copy(),
            joint_position=data.joint_pos.torch[0].cpu().numpy().copy(),
            actions=actions[0].cpu().numpy().copy(),
            foot_position_w=data.body_link_pose_w.torch[0, foot_ids, :3].cpu().numpy().copy(),
            route_position=route,
            root_height_m=height,
            up_cosine=up,
            air_drag_force_n=env.air_drag.force_w[0].sum(0).cpu().numpy().copy(),
            min_soil_displacement_m=minimum,
            max_soil_displacement_m=maximum,
        )
        for key, value in values.items():
            assert np.isfinite(value).all(), f"Nonfinite {key}"
            traces[key].append(value)
        x, y = position[:2]
        # A 1 m guard contains the yaw-rotated 1.6 x 1 m scanner and the feet.
        inside = (-1 < x < 1 and -6 < y < 0.5) if condition == "earth" else (-0.5 < x < 6 and -1 < y < 1)
        outcome = classify_sample(position[:2], target, height, up, values["time_s"], inside, protocol)
        if outcome:
            break
        if step and step % 250 == 0:
            print(
                f"TRIAL_PROGRESS condition={condition} seed={sample['seed']} seconds={step * env.step_dt:.2f} "
                f"forward_m={route[0]:.3f} lateral_m={route[1]:.3f}",
                flush=True,
            )
    assert outcome is not None
    arrays = {k: np.asarray(v) for k, v in traces.items()}
    path = output / f"trial-{sample['seed']:06d}.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    result = dict(
        condition=condition,
        seed=sample["seed"],
        perturbation=sample,
        outcome=outcome,
        duration_s=float(arrays["time_s"][-1]),
        start_position_w=start.tolist(),
        target_xy=target.tolist(),
        initial_velocity_w=initial_velocity.tolist(),
        initial_quaternion_xyzw=arrays["root_quaternion_xyzw"][0].tolist(),
        final_position_w=position.tolist(),
        final_forward_m=float(route[0]),
        final_lateral_m=float(route[1]),
        max_abs_lateral_m=float(np.abs(arrays["route_position"][:, 1]).max()),
        final_goal_error_m=float(np.linalg.norm(position[:2] - target)),
        path_length_m=float(np.linalg.norm(np.diff(arrays["root_position_w"][:, :2], axis=0), axis=1).sum()),
        minimum_root_height_m=float(arrays["root_height_m"].min()),
        minimum_up_cosine=float(arrays["up_cosine"].min()),
        maximum_soil_mass_error_kg=max_mass_error,
        soil_reset_verified=condition == "mars",
        samples=len(arrays["time_s"]),
        trajectory_file=path.name,
        wall_seconds=time.monotonic() - t0,
    )
    with path.with_suffix(".json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=["earth", "mars"], required=True)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default=".pretrained_checkpoints/h1_rough.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    add_launcher_args(parser)
    parser.set_defaults(visualizer="none")
    args = parser.parse_args()
    if args.trials < 1 or args.seed < 0:
        parser.error("trials must be positive and seed nonnegative")
    if args.visualizer not in ("none", [], ["none"]):
        parser.error("Evaluation is headless; use view_scene.py for captures")
    import gymnasium as gym
    import torch

    from mars_project.environment.mpm_soil import SOIL_MATERIAL, THICKNESS, VOXEL_SIZE, verify_soil_loaded
    from mars_project.scene_runner import load_preview_policy

    cfg, yaw = configure_evaluation(args.condition)
    output = args.output_dir.resolve() / args.condition
    output.mkdir(parents=True, exist_ok=False)
    protocol = Protocol()
    manifest = dict(
        condition=args.condition,
        requested_trials=args.trials,
        seeds=list(range(args.seed, args.seed + args.trials)),
        protocol=protocol.to_dict(),
        route_yaw_rad=yaw,
        physics_dt=cfg.sim.dt,
        control_dt=cfg.sim.dt * cfg.decimation,
        backend="Newton/MuJoCo Warp",
        soil_coupler=args.condition == "mars",
        rigid_substeps=2,
        rigid_contacts="Newton collision pipeline",
        checkpoint=str(Path(args.checkpoint).resolve()),
        checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted([*Path("src/mars_project").rglob("*.py"), Path(__file__)])
        },
        soil_material=SOIL_MATERIAL if args.condition == "mars" else None,
        soil_thickness_m=THICKNESS if args.condition == "mars" else None,
        soil_grid_m=VOXEL_SIZE if args.condition == "mars" else None,
        limitations=[
            "Different source terrains: this compares combined environments, not isolated gravity effects.",
            "MPM material is assumed, not calibrated; convergence thresholds remain unresolved.",
            "Small initial-state perturbations sample a narrow envelope, not broad reliability.",
            "Fall = base below 0.55 m above local ground or tilt above 60 degrees; no contact-based fall test.",
            "Boundary exits are failures of the bounded route; they are not falls.",
            "No contact-based slip metric: foot positions are saved but not called slip measurements.",
        ],
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    results = []
    with launch_simulation(cfg, args):
        env = gym.make("Mars-Project-H1", cfg=cfg)
        try:
            env.reset()
            if args.condition == "mars":
                verify_soil_loaded(env.unwrapped)
            wrapped, policy, _ = load_preview_policy(env, args.checkpoint)
            manifest["joint_names"] = env.unwrapped.scene["robot"].joint_names
            manifest["body_names"] = env.unwrapped.scene["robot"].body_names
            manifest["air_density_kg_m3"] = env.unwrapped.air_drag.density
            manifest["air_force_refresh_s"] = (
                env.unwrapped.step_dt
                if env.unwrapped.sim.physics_manager.handles_decimation()
                else env.unwrapped.physics_dt
            )
            manifest["asset_sha256"] = {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path("src/mars_project/assets/data").rglob("*ground*")
                if p.is_file()
            }
            (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            with torch.inference_mode():
                for seed in manifest["seeds"]:
                    result = run_trial(
                        env.unwrapped, wrapped, policy, args.condition, paired_start(seed), yaw, output, protocol
                    )
                    results.append(result)
                    print("TRIAL " + json.dumps(result, allow_nan=False), flush=True)
            summary = summarize(results)
            summary["complete"] = len(results) == args.trials
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print("EVALUATION_COMPLETE " + json.dumps(summary), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
