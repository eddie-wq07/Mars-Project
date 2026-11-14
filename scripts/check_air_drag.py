"""Verify drag changes H1 momentum, composes with other loads, and resets cleanly.

    python scripts/check_air_drag.py --backend newton_mjwarp
    python scripts/check_air_drag.py --backend isaacsim_physx --visualizer kit --kit_args=--no-window

Two free-flying robots start at 1 m/s in zero gravity: Earth air vs no drag.
With --soil mpm, check one free-flying robot over the coupled soil instead.
This is a force-integration check, not a locomotion experiment.
"""

import argparse
import json

from isaaclab.app import add_launcher_args, launch_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["newton_mjwarp", "isaacsim_physx"], default="newton_mjwarp")
    parser.add_argument("--decimation", type=int, choices=[1, 4], default=4)
    parser.add_argument("--soil", choices=["rigid", "mpm"], default="rigid")
    add_launcher_args(parser)
    parser.set_defaults(visualizer="none")
    args = parser.parse_args()
    if args.soil == "mpm" and args.backend != "newton_mjwarp":
        parser.error("MPM requires the Newton backend")

    import gymnasium as gym
    import torch

    from isaaclab_tasks.utils import parse_env_cfg

    import mars_project.tasks  # noqa: F401
    from mars_project.environment.aerodynamics import air_density
    from mars_project.scenes.setup import configure_preview

    num_envs = 1 if args.soil == "mpm" else 2
    cfg = parse_env_cfg("Mars-Project-H1", num_envs=num_envs, overrides=[f"physics={args.backend}"])
    configure_preview(cfg, "grand-canyon", soil=args.soil, air_drag="earth")
    cfg.sim.gravity = (0.0, 0.0, 0.0)
    cfg.decimation = args.decimation
    cfg.scene.robot.init_state.pos = (0.0, 0.0, 5.0)
    cfg.scene.robot.init_state.lin_vel = (1.0, 0.0, 0.0)
    if args.soil == "mpm":
        # Fall detection uses the gravity direction, undefined in this zero-g check.
        cfg.terminations.soil_fall = None
    cfg.ui_window_class_type = None
    with launch_simulation(cfg, args):
        env = gym.make("Mars-Project-H1", cfg=cfg).unwrapped
        try:
            env.reset()
            assert env.action_manager.total_action_dim == 19, "Pretrained action space changed"
            robot = env.scene["robot"]
            drag = env.air_drag
            assert drag.dimensions.shape == (robot.num_bodies, 3)
            # Only this diagnostic uses per-environment densities.
            drag.density = torch.tensor([air_density("earth"), 0.0][:num_envs], device=env.device).reshape(
                num_envs, 1, 1
            )
            mass = robot.data.body_mass.torch.to(env.device)[..., None]

            def momentum():
                return (mass * robot.data.body_com_vel_w.torch[..., :3]).sum(1).clone()

            p_initial = momentum()
            extra_force = torch.zeros_like(drag.force_w)
            extra_force[:, robot.body_names.index("torso_link"), 1] = 1.0
            robot.permanent_wrench_composer.add_forces_and_torques_index(forces=extra_force, is_global=True)
            impulse = torch.zeros((num_envs, 3), device=env.device)
            actions = torch.zeros((num_envs, 19), device=env.device)
            samples = []
            apply_drag = drag.apply_actions

            def sample_force():
                apply_drag()
                samples.append((drag.force_w + extra_force).sum(1).clone())

            drag.apply_actions = sample_force
            steps = round(0.5 / env.step_dt)
            with torch.inference_mode():
                for _ in range(steps):
                    samples.clear()
                    _, _, terminated, truncated, _ = env.step(actions)
                    assert not (terminated | truncated).any(), "Unexpected free-flight reset"
                    impulse += torch.stack(samples).mean(0) * env.step_dt
            refresh_dt = env.step_dt / len(samples)
            drag.apply_actions = apply_drag
            measured = momentum() - p_initial
            error = (measured - impulse).abs().max().item()
            assert error < 0.003, f"Momentum did not follow applied impulse: {measured} vs {impulse}"
            assert measured[0, 0] < -0.05, "Earth drag did not slow H1"
            if num_envs == 2:
                assert abs(measured[1, 0]) < 0.003, "Disabled drag changed forward momentum"
            assert measured[:, 1].min() > 0.45, "Drag overwrote the independent 1 N load"
            env.reset()
            assert torch.count_nonzero(drag.force_w) == 0, "Drag telemetry survived reset"
            drag.density = 0.0
            reset_momentum = momentum()
            for _ in range(10):
                env.step(actions)
            assert (momentum() - reset_momentum).abs().max() < 0.003, "Stale external force survived reset"
            print(
                "AIR_DRAG_CHECK "
                + json.dumps(
                    dict(
                        backend=args.backend,
                        soil=args.soil,
                        passed=True,
                        action_dim=19,
                        duration_s=steps * env.step_dt,
                        decimation=cfg.decimation,
                        force_refresh_s=refresh_dt,
                        measured_impulse_Ns=measured.cpu().tolist(),
                        expected_impulse_Ns=impulse.cpu().tolist(),
                        max_error_Ns=error,
                        additive_load_verified=True,
                        reset_verified=True,
                    )
                ),
                flush=True,
            )
        finally:
            env.close()


if __name__ == "__main__":
    main()
