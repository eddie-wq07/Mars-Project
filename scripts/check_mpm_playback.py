"""Check loaded H1/MPM state, interaction, live sensing, visualization and reset.

This is an integration check, not soil material or locomotion validation.
Use --visualizer kit --kit_args=--no-window to additionally check rendered USD updates.
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_launcher_args(parser)
    parser.set_defaults(visualizer="none")
    args = parser.parse_args()

    import gymnasium as gym
    import numpy as np
    import torch

    from isaaclab.managers import SceneEntityCfg

    from isaaclab_tasks.utils import parse_env_cfg

    import mars_project.tasks  # noqa: F401
    from mars_project.environment.mpm_soil import mpm_height_scan, soil_progress, verify_soil_loaded
    from mars_project.scenes.setup import configure_preview

    cfg = parse_env_cfg("Mars-Project-H1", num_envs=1, overrides=["physics=newton_mjwarp"])
    configure_preview(cfg, "grand-canyon")
    cfg.ui_window_class_type = None
    with launch_simulation(cfg, args):
        env = gym.make("Mars-Project-H1", cfg=cfg).unwrapped
        try:
            env.reset()
            verify_soil_loaded(env)
            soil = env.scene["soil"]
            initial = soil.data.particle_pos_w.torch.clone()
            runtime = env.soil_runtime
            untouched = torch.linalg.vector_norm(initial[0, :, :2], dim=-1) > 2.0
            assert int(runtime.dynamic.sum()) < len(initial[0]) // 4
            sensor = env.scene["height_scanner"]
            rigid_scan = sensor.data.pos_w.torch[:, 2, None] - sensor.data.ray_hits_w.torch[..., 2] - 0.5
            live_scan = mpm_height_scan(env, SceneEntityCfg("height_scanner"))
            assert torch.isfinite(live_scan).all()
            assert torch.max(rigid_scan - live_scan) > 0.1, "Policy does not see the MPM layer"
            zeros = torch.zeros((1, env.action_manager.total_action_dim), device=env.device)
            peak_force = 0.0
            with torch.inference_mode():
                for step in range(50):
                    obs, _, terminated, truncated, _ = env.step(zeros)
                    assert not (terminated | truncated).any(), f"Unexpected reset at control step {step + 1}"
                    assert torch.isfinite(obs["policy"]).all()
                    assert soil.data.particle_pos_w.torch.isfinite().all()
                    assert torch.equal(soil.data.particle_pos_w.torch[0, untouched], initial[0, untouched]), (
                        "Undisturbed material was advanced by the solver"
                    )
                    if (step + 1) % 10 == 0:
                        force = soil_progress(env)
                        peak_force = max(peak_force, float(force[2]))
            displacement = float((soil.data.particle_pos_w.torch - initial).abs().max())
            assert displacement > 1e-4, "Loaded particles did not move"
            assert peak_force > 1.0, "No soil reaction on H1 was measured"
            if env.sim.is_rendering:
                for _ in range(3):
                    env.sim.render()
                surface = runtime.surface
                assert surface is not None
                rendered = np.asarray(surface.mesh.GetPointsAttr().Get())
                assert np.allclose(
                    rendered[:, 2] - surface.base[:, 2],
                    runtime.displacement_at(surface.query).cpu().numpy(),
                    atol=1e-5,
                ), "Rendered soil did not follow solver state"
                assert not env.sim.stage.GetPrimAtPath("/World/envs/env_0/Soil/Particles")
            env.reset()
            assert torch.allclose(soil.data.particle_pos_w.torch, initial, atol=1e-6), "Soil reset failed"
            assert torch.count_nonzero(soil.data.particle_vel_w.torch) == 0
            assert torch.count_nonzero(runtime.displacement) == 0
            assert runtime.dust.emitted == 0
            # Exercise a solver step after clearing the particle/coupling state.
            obs, _, _, _, _ = env.step(zeros)
            assert torch.isfinite(obs["policy"]).all()
            print(
                f"MPM_PLAYBACK_CHECK passed particles={soil.particles_per_object} "
                f"peak_robot_soil_force_z_n={peak_force:.4f} max_particle_motion_m={displacement:.6f} "
                f"height_scan=live reset=passed visual_sync={'passed' if env.sim.is_rendering else 'not_run'} "
                "material_validated=False",
                flush=True,
            )
        finally:
            env.close()


if __name__ == "__main__":
    main()
