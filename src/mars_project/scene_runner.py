"""H1 scene preview with optional pretrained-policy inference and a fixed camera."""

import argparse
import subprocess
import sys
from pathlib import Path

import warp as wp

from isaaclab.app import add_launcher_args, launch_simulation

wp.config.enable_backward = False


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gravity",
        choices=["earth", "mars"],
        default=None,
        help="Default: Earth for earth appearance, Mars for other appearances",
    )
    parser.add_argument(
        "--appearance",
        choices=["earth", "grand-canyon"],
        default="earth",
        help="Visual backdrop only; independent of gravity",
    )
    parser.add_argument("--steps", type=int, default=0, help="0 runs until the window closes")
    parser.add_argument(
        "--checkpoint", help="Optional trained rough-H1 RSL-RL checkpoint; otherwise zero-action preview"
    )
    parser.add_argument(
        "--screenshot",
        nargs="?",
        const=True,
        default=None,
        type=str,
        help="Capture a timestamped PNG after --steps; optionally supply a descriptive label",
    )
    parser.add_argument(
        "--screenshot-pair",
        metavar="LABEL",
        help="Capture LABEL-A after --steps and LABEL-B one simulation second later, in a LABEL subfolder",
    )
    parser.add_argument(
        "--screenshot-series", metavar="LABEL", help="Capture a series one simulation second apart in a LABEL folder"
    )
    parser.add_argument("--screenshot-count", type=int, default=5, help="Series length (1–26); default 5")
    parser.add_argument(
        "--screenshot-requested",
        "--capture-requested",
        action="store_true",
        help="Mark screenshots as explicitly requested by the user; otherwise save under date/misc",
    )
    parser.add_argument("--record", metavar="LABEL", help="Record --steps as a 25-fps MP4 in a LABEL folder")
    parser.add_argument(
        "--camera-motion", choices=["fixed", "orbit"], default="fixed", help="Camera motion for MP4 recording"
    )
    add_launcher_args(parser)
    parser.set_defaults(visualizer="kit")
    args = parser.parse_args()
    if args.gravity is None:
        args.gravity = "earth" if args.appearance == "earth" else "mars"
    if (args.screenshot or args.screenshot_pair or args.screenshot_series or args.record) and args.steps <= 0:
        parser.error("--screenshot requires --steps greater than zero")
    if sum(bool(value) for value in (args.screenshot, args.screenshot_pair, args.screenshot_series, args.record)) > 1:
        parser.error("choose one capture mode: --screenshot, --screenshot-pair, --screenshot-series, --record")
    if not 1 <= args.screenshot_count <= 26:
        parser.error("--screenshot-count must be between 1 and 26")
    if args.camera_motion != "fixed" and not args.record:
        parser.error("--camera-motion requires --record")
    if args.checkpoint and not Path(args.checkpoint).expanduser().is_file():
        parser.error("checkpoint file does not exist")
    return args


def main():
    args = parse_args()
    # Keep standalone USD dependency inspection out of Kit's process.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from mars_project.assets.scene_paths import validate_assets; validate_assets(sys.argv[2])",
            str(Path(__file__).resolve().parents[1]),
            args.appearance,
        ],
        check=True,
    )
    import gymnasium as gym
    import torch

    from isaaclab_tasks.utils import parse_env_cfg

    import mars_project.tasks  # noqa: F401 -- registers Mars-Project-H1
    from mars_project.scenes.setup import configure_preview

    cfg = parse_env_cfg("Mars-Project-H1", num_envs=1, overrides=["physics=isaacsim_physx"])
    cfg.sim.gravity = (0.0, 0.0, -3.71 if args.gravity == "mars" else -9.81)
    configure_preview(cfg, args.appearance)
    with launch_simulation(cfg, args):
        print(f"Creating environment with {args.appearance} appearance...", flush=True)
        env = gym.make("Mars-Project-H1", cfg=cfg)
        try:
            if args.appearance == "earth":
                from mars_project.scenes.earth import add_earth_scene, apply_earth_render_settings

                apply_earth_render_settings()
                # Let Kit consume initial-view USD/settings changes before reset/camera switching.
                for _ in range(3):
                    env.unwrapped.sim.render()
                add_earth_scene()
            if args.appearance == "grand-canyon":
                from mars_project.scenes.canyon import add_canyon_scene

                add_canyon_scene()
            env.reset()
            robot_data = env.unwrapped.scene["robot"].data
            print(
                f"INITIAL_ROBOT position={robot_data.root_pos_w.torch[0].tolist()} "
                f"quaternion_xyzw={robot_data.root_quat_w.torch[0].tolist()} "
                f"velocity={robot_data.root_vel_w.torch[0].tolist()}",
                flush=True,
            )
            import carb.settings

            carb.settings.get_settings().set("/rtx/rendermode", "RealTimePathTracing")
            from omni.kit.viewport.utility import get_active_viewport

            from isaaclab.sim.utils.stage import get_current_stage

            from mars_project.scenes.cameras import apply_view_settings, create_camera

            stage = get_current_stage()
            camera_path = create_camera(stage, cfg)
            viewport = get_active_viewport()
            if viewport:
                viewport.set_active_camera(camera_path)
                apply_view_settings(stage, camera_path, viewport.render_product_path)
            policy = None
            observations = None
            if args.checkpoint:
                from isaaclab_rl.rsl_rl import (
                    RslRlVecEnvWrapper,
                    check_rsl_rl_version,
                    create_rsl_rl_runner,
                    handle_deprecated_rsl_rl_cfg,
                )

                from isaaclab_tasks.utils import load_cfg_from_registry

                agent_cfg = load_cfg_from_registry("Mars-Project-H1", "rsl_rl_cfg_entry_point")
                # Use Isaac Lab's version-aware migration before constructing RSL-RL models.
                agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
                agent_cfg.device = env.unwrapped.device
                env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
                runner = create_rsl_rl_runner(env, agent_cfg)
                runner.load(str(Path(args.checkpoint).expanduser().resolve()))
                policy = runner.get_inference_policy(device=env.unwrapped.device)
                observations = env.get_observations()

            def action_step():
                nonlocal observations
                if policy is None:
                    env.step(
                        torch.zeros(
                            (env.unwrapped.num_envs, env.unwrapped.action_manager.total_action_dim),
                            device=env.unwrapped.device,
                        )
                    )
                else:
                    observations, _, dones, _ = env.step(policy(observations))
                    policy.reset(dones)

            print(
                f"H1 scene ready: {args.gravity}, gravity={cfg.sim.gravity}. "
                + ("Checkpoint loaded." if policy else "Zero-action preview; no trained policy.")
            )
            step = 0
            with torch.inference_mode():
                if args.record:
                    from mars_project.recordings import record_video

                    record_video(
                        env.unwrapped.sim,
                        env.unwrapped.scene["robot"],
                        action_step,
                        args.steps,
                        env.unwrapped.step_dt,
                        args.record,
                        requested=args.screenshot_requested,
                        camera_motion=args.camera_motion,
                        camera_target=(
                            cfg.scene.robot.init_state.pos[0],
                            cfg.scene.robot.init_state.pos[1],
                            cfg.scene.robot.init_state.pos[2] - 0.1,
                        ),
                    )
                    step = args.steps
                else:
                    while env.unwrapped.sim.is_running() and (not args.steps or step < args.steps):
                        action_step()
                        step += 1
            if args.screenshot_pair or args.screenshot_series:
                from mars_project.screenshots import capture_screenshot

                label = args.screenshot_pair or args.screenshot_series
                count = 2 if args.screenshot_pair else args.screenshot_count
                interval_steps = round(1.0 / env.unwrapped.step_dt)
                if abs(interval_steps * env.unwrapped.step_dt - 1.0) > 1e-6:
                    raise ValueError("One second must be an integer number of control steps")
                for index in range(count):
                    if index:
                        with torch.inference_mode():
                            for _ in range(interval_steps):
                                action_step()
                                step += 1
                    capture_screenshot(
                        env.unwrapped.sim,
                        env.unwrapped.scene["robot"],
                        f"{label}-{chr(ord('A') + index)}",
                        subfolder=label,
                        requested=args.screenshot_requested,
                        reframe=index == 0,
                    )
                    print(f"CAPTURE_STEP step={step} simulation_seconds={step * env.unwrapped.step_dt:.3f}", flush=True)
                print(
                    f"SCREENSHOT_SERIES count={count} interval_seconds=1.0 interval_steps={interval_steps}", flush=True
                )
            if args.screenshot:
                from mars_project.screenshots import capture_screenshot

                capture_screenshot(
                    env.unwrapped.sim,
                    env.unwrapped.scene["robot"],
                    args.appearance if args.screenshot is True else args.screenshot,
                    requested=args.screenshot_requested,
                )
            if args.steps:
                print(f"PREVIEW_OK steps={step}")
        finally:
            env.close()


if __name__ == "__main__":
    main()
