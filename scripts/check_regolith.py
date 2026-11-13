"""No-policy PhysX check: drop a loaded foot, sweep it, and verify grain displacement."""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_launcher_args(parser)
    parser.set_defaults(visualizer="kit")
    args = parser.parse_args()
    import gymnasium as gym
    import torch

    import isaaclab.sim as sim
    from isaaclab.assets import RigidObjectCfg

    from isaaclab_tasks.utils import parse_env_cfg

    import mars_project.tasks  # noqa: F401
    from mars_project.environment.regolith import GRAIN_RADIUS, surface_heights
    from mars_project.scenes.setup import configure_preview

    cfg = parse_env_cfg("Mars-Project-H1", num_envs=1, overrides=["physics=isaacsim_physx"])
    configure_preview(cfg, "grand-canyon")
    cfg.scene.robot.init_state.pos = (-1.0, 1.2, 1.05)
    cfg.scene.probe = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Probe",
        spawn=sim.CuboidCfg(
            size=(0.25, 0.12, 0.04),
            rigid_props=sim.RigidBodyPropertiesCfg(),
            mass_props=sim.MassPropertiesCfg(mass=15),
            collision_props=sim.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0),
            physics_material=sim.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6, restitution=0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0, 0.25)),
    )
    # Robot failure must not reset the independent soil experiment.
    cfg.terminations.base_contact = None
    with launch_simulation(cfg, args):
        env = gym.make("Mars-Project-H1", cfg=cfg).unwrapped
        try:
            env.reset()
            grains = env.scene["regolith"]
            probe = env.scene["probe"]
            initial = grains.data.body_link_pos_w.torch.clone()
            zeros = torch.zeros((1, env.action_manager.total_action_dim), device=env.device)
            for _ in range(100):
                env.step(zeros)
            settled = grains.data.body_link_pos_w.torch.clone()
            foot_z = probe.data.root_pos_w.torch[0, 2].item()
            assert 0.02 < foot_z < 0.15, foot_z
            for _ in range(100):
                velocity = probe.data.root_vel_w.torch.clone()
                velocity[:, 0] = 0.3
                probe.write_root_velocity_to_sim(velocity)
                env.step(zeros)
            final = grains.data.body_link_pos_w.torch.clone()
            movement = torch.linalg.vector_norm(final[..., :2] - settled[..., :2], dim=-1)
            moved = int((movement > 0.005).sum())
            assert moved > 5, (moved, float(movement.max()))
            far = settled[..., 0] > 2.0
            far_shift = movement[far].median().item()
            assert far_shift < 0.01, f"Unloaded bed drifts too much: {far_shift}"
            assert torch.isfinite(final).all()
            assert (final[..., 2] > -0.1).all(), "Grains penetrated the terrain"
            hits = final[:, :5].clone()
            hits[..., 2] = 0
            surface = surface_heights(hits, final)
            assert torch.all(surface >= final[:, :5, 2] + GRAIN_RADIUS - 1e-5)
            env.reset()
            restored = grains.data.body_link_pos_w.torch
            assert torch.allclose(initial, restored, atol=1e-5), "Reset did not restore soil"
            print(
                f"REGOLITH_CHECK passed grains={initial.shape[1]} foot_z={foot_z:.4f} "
                f"moved={moved} max_shift={movement.max().item():.4f} far_shift={far_shift:.4f}",
                flush=True,
            )
        finally:
            env.close()


if __name__ == "__main__":
    main()
