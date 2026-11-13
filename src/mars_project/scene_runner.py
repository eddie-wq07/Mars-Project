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
    add_launcher_args(parser)
    parser.set_defaults(visualizer="kit")
    args = parser.parse_args()
    if args.gravity is None:
        args.gravity = "earth" if args.appearance == "earth" else "mars"
    if (args.screenshot or args.screenshot_pair) and args.steps <= 0:
        parser.error("--screenshot requires --steps greater than zero")
    if args.screenshot and args.screenshot_pair:
        parser.error("choose --screenshot or --screenshot-pair")
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
                add_earth_scene()
            if args.appearance == "grand-canyon":
                from mars_project.scenes.canyon import add_canyon_scene

                add_canyon_scene()
            env.reset()
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
                while env.unwrapped.sim.is_running() and (not args.steps or step < args.steps):
                    action_step()
                    step += 1
            if args.screenshot_pair:
                from mars_project.screenshots import capture_screenshot

                interval_steps = round(1.0 / env.unwrapped.step_dt)
                if abs(interval_steps * env.unwrapped.step_dt - 1.0) > 1e-6:
                    raise ValueError("One second must be an integer number of control steps")
                capture_screenshot(
                    env.unwrapped.sim,
                    env.unwrapped.scene["robot"],
                    f"{args.screenshot_pair}-A",
                    subfolder=args.screenshot_pair,
                )
                with torch.inference_mode():
                    for _ in range(interval_steps):
                        action_step()
                        step += 1
                capture_screenshot(
                    env.unwrapped.sim,
                    env.unwrapped.scene["robot"],
                    f"{args.screenshot_pair}-B",
                    subfolder=args.screenshot_pair,
                    reframe=False,
                )
                print(f"SCREENSHOT_PAIR interval_seconds=1.0 interval_steps={interval_steps}", flush=True)
            if args.screenshot:
                from mars_project.screenshots import capture_screenshot

                capture_screenshot(
                    env.unwrapped.sim,
                    env.unwrapped.scene["robot"],
                    args.appearance if args.screenshot is True else args.screenshot,
                )
            if args.steps:
                print(f"PREVIEW_OK steps={step}")
        finally:
            env.close()


if __name__ == "__main__":
    main()
