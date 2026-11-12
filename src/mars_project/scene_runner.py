"""Standalone H1 scene: default joint-position targets, not a trained walking policy."""
import argparse
from pathlib import Path

import warp as wp
wp.config.enable_backward = False

from isaaclab.app import add_launcher_args, launch_simulation

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gravity', choices=['earth', 'mars'], default=None,
                        help='Default: Earth for earth appearance, Mars for other appearances')
    parser.add_argument('--appearance',
                        type=lambda value: value.replace('_', '-').replace('main-creator', 'main-crater'),
                        choices=['earth', 'grand-canyon'], default='earth',
                        help='Visual backdrop only; independent of gravity')
    parser.add_argument('--steps', type=int, default=0, help='0 runs until the window closes')
    parser.add_argument('--view', choices=['cinematic', 'robot'], default='cinematic')
    parser.add_argument('--camera-motion', choices=['fixed', 'orbit'], default='fixed')
    parser.add_argument('--orbit-radius', type=float, default=3., help='Orbit radius around initial torso position, meters')
    parser.add_argument('--orbit-height', type=float, default=.8, help='Camera height above orbit target, meters')
    parser.add_argument('--orbit-degrees', type=float, default=30.)
    parser.add_argument('--orbit-duration', type=float, default=5., help='Seconds for the eased orbit')
    parser.add_argument('--pov-offset', type=float, nargs=3, default=(0.22, 0.0, 0.35),
                        help='Virtual camera XYZ offset from H1 torso_link, in meters')
    parser.add_argument('--checkpoint', help='Optional trained flat-H1 RSL-RL checkpoint; otherwise zero-action preview')
    add_launcher_args(parser)
    parser.set_defaults(visualizer="kit")
    args = parser.parse_args()
    if args.gravity is None:
        args.gravity = 'earth' if args.appearance == 'earth' else 'mars'
    if args.orbit_duration <= 0 or args.orbit_radius <= 0:
        parser.error('orbit-duration and orbit-radius must be positive')
    if args.checkpoint and not Path(args.checkpoint).expanduser().is_file():
        parser.error('checkpoint file does not exist')
    return args


def main():
    args = parse_args()
    from mars_project.assets.scene_paths import validate_assets
    validate_assets(args.appearance)
    import gymnasium as gym
    import torch
    import isaaclab_tasks  # registers official tasks
    import mars_project.tasks.h1
    from mars_project.scenes.setup import configure_preview
    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg('Mars-Project-H1', num_envs=1,
                        overrides=['physics=isaacsim_physx'])
    cfg.sim.gravity = (0., 0., -3.71 if args.gravity == 'mars' else -9.81)
    configure_preview(cfg, args.appearance)
    with launch_simulation(cfg, args):
        print(f'Creating environment with {args.appearance} appearance...', flush=True)
        env = gym.make('Mars-Project-H1', cfg=cfg)
        try:
            if args.appearance == 'earth':
                from mars_project.scenes.earth import add_earth_scene, apply_earth_render_settings
                apply_earth_render_settings()
                add_earth_scene()
            if args.appearance == 'grand-canyon':
                from mars_project.scenes.canyon import add_canyon_scene
                add_canyon_scene(args.appearance)
            env.reset()
            import carb.settings
            carb.settings.get_settings().set('/rtx/rendermode', 'RealTimePathTracing')
            from isaaclab.sim.utils.stage import get_current_stage
            from omni.kit.viewport.utility import get_active_viewport
            from mars_project.scenes.cameras import create_cameras, update_robot_camera, update_cinematic_camera
            stage = get_current_stage()
            cameras = create_cameras(stage, cfg, args.pov_offset)
            robot = env.unwrapped.scene['robot']
            torso_index = robot.body_names.index('torso_link')
            orbit_center = robot.data.body_link_pose_w.torch[0, torso_index, :3].detach().cpu().tolist()
            def update_cinematic(seconds):
                update_cinematic_camera(stage, cameras['cinematic'], cfg, seconds, args.orbit_duration,
                                        args.orbit_degrees, orbit_center, args.orbit_radius, args.orbit_height)
            if args.camera_motion == 'orbit':
                update_cinematic(0.)
            def update_pov():
                update_robot_camera(stage, robot, cameras['robot'], args.pov_offset)
            update_pov()
            viewport = get_active_viewport()
            if viewport:
                viewport.set_active_camera(cameras[args.view])
                if args.appearance == 'earth':
                    from mars_project.scenes.earth import author_earth_view_settings
                    author_earth_view_settings(stage, cameras[args.view], viewport.render_product_path)
            policy = None
            observations = None
            if args.checkpoint:
                from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner
                from isaaclab_tasks.utils import load_cfg_from_registry
                agent_cfg = load_cfg_from_registry('Mars-Project-H1', 'rsl_rl_cfg_entry_point')
                env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
                runner = create_rsl_rl_runner(env, agent_cfg)
                runner.load(str(Path(args.checkpoint).expanduser().resolve()))
                policy = runner.get_inference_policy(device=env.unwrapped.device)
                observations = env.get_observations()
            def action_step():
                nonlocal observations
                if policy is None:
                    env.step(torch.zeros((env.unwrapped.num_envs, env.unwrapped.action_manager.total_action_dim),
                                         device=env.unwrapped.device))
                else:
                    observations, _, dones, _ = env.step(policy(observations))
                    policy.reset(dones)
                update_pov()
            print(f'H1 scene ready: {args.gravity}, gravity={cfg.sim.gravity}. ' +
                  ('Checkpoint loaded.' if policy else 'Zero-action preview; no trained policy.'))
            step = 0
            with torch.inference_mode():
                while env.unwrapped.sim.is_running() and (not args.steps or step < args.steps):
                    if args.camera_motion == 'orbit':
                        update_cinematic(step*env.unwrapped.step_dt)
                    action_step()
                    step += 1
            if args.steps:
                print(f'PREVIEW_OK steps={step}')
        finally:
            env.close()


if __name__ == '__main__':
    main()
