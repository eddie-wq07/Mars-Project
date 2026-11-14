"""Check seven seconds of the actual preview policy before expensive video rendering."""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=".pretrained_checkpoints/h1_rough.pt")
    parser.add_argument("--seconds", type=float, default=7.0)
    add_launcher_args(parser)
    parser.set_defaults(visualizer="none")
    args = parser.parse_args()
    import gymnasium as gym
    import torch

    from isaaclab_tasks.utils import parse_env_cfg

    import mars_project.tasks  # noqa: F401
    from mars_project.environment.mpm_soil import soil_progress, verify_soil_loaded
    from mars_project.environment.resistance import apply_joint_resistance, environment_profile
    from mars_project.scene_runner import load_preview_policy
    from mars_project.scenes.setup import configure_preview

    cfg = parse_env_cfg("Mars-Project-H1", num_envs=1, overrides=["physics=newton_mjwarp"])
    cfg.sim.gravity = (0.0, 0.0, -3.71)
    configure_preview(cfg, "grand-canyon", 0.5, soil="mpm")
    apply_joint_resistance(cfg, environment_profile("mars"), 0.0)
    cfg.ui_window_class_type = None
    with launch_simulation(cfg, args):
        env = gym.make("Mars-Project-H1", cfg=cfg)
        try:
            env.reset()
            verify_soil_loaded(env.unwrapped)
            env, policy, observations = load_preview_policy(env, args.checkpoint)
            runtime = env.unwrapped.soil_runtime
            initial_mass = float(runtime.mass.double().sum())
            with torch.inference_mode():
                for step in range(round(args.seconds / env.unwrapped.step_dt)):
                    observations, _, dones, _ = env.step(policy(observations))
                    assert not dones.any(), f"Policy reset at step {step}"
                    assert all(torch.isfinite(value).all() for value in observations.values())
                    mass = float(runtime.mass.double().sum()) + runtime.dust.mass.sum() + runtime.dust.deposited
                    assert abs(mass - initial_mass) < 1e-5, f"Soil/dust mass drift: {mass - initial_mass} kg"
                    if (step + 1) % 50 == 0:
                        soil_progress(env.unwrapped)
                        print(
                            f"SOIL_WALK seconds={(step + 1) * env.unwrapped.step_dt:.2f} "
                            f"position={env.unwrapped.scene['robot'].data.root_pos_w.torch[0].tolist()} resets=0",
                            flush=True,
                        )
            assert runtime.dust.emitted > 0, "Walking did not exercise the dust transfer path"
            print(f"SOIL_WALK_CHECK passed seconds={args.seconds} mass_conserved=True dust_emitted=True", flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
