"""Legacy prescribed-deformation check; this does NOT validate soil mechanics."""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-heuristic", action="store_true", help="Reproduce the retired kinematic-column test")
    add_launcher_args(parser)
    parser.set_defaults(visualizer="kit")
    args = parser.parse_args()
    if not args.legacy_heuristic:
        parser.error("Retired heuristic test. Use scripts/verify_mpm_soil.py for solver verification.")
    import gymnasium as gym
    import numpy as np
    import torch

    import isaaclab.sim as sim
    from isaaclab.assets import RigidObjectCfg

    from isaaclab_tasks.utils import parse_env_cfg

    import mars_project.tasks  # noqa: F401
    from mars_project.environment.regolith import CELL, COLUMN_DEPTH, configure_regolith
    from mars_project.scenes.setup import configure_preview

    cfg = parse_env_cfg("Mars-Project-H1", num_envs=1, overrides=["physics=isaacsim_physx"])
    configure_preview(cfg, "grand-canyon", soil="rigid")
    configure_regolith(cfg)
    cfg.scene.robot.init_state.pos = (-1.0, 1.5, 1.05)
    cfg.scene.probe = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Probe",
        spawn=sim.CuboidCfg(
            size=(0.25, 0.12, 0.04),
            rigid_props=sim.RigidBodyPropertiesCfg(),
            mass_props=sim.MassPropertiesCfg(mass=15),
            collision_props=sim.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0),
            physics_material=sim.RigidBodyMaterialCfg(static_friction=0.65, dynamic_friction=0.45, restitution=0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0, 0.25)),
    )
    cfg.terminations.base_contact = None
    with launch_simulation(cfg, args):
        env = gym.make("Mars-Project-H1", cfg=cfg).unwrapped
        try:
            env.reset()
            probe = env.scene["probe"]
            sand = env.sand
            zeros = torch.zeros((1, env.action_manager.total_action_dim), device=env.device)
            for _ in range(50):
                env.step(zeros)
            before = probe.data.root_pos_w.torch[0, 2].item()
            for _ in range(75):
                foot = probe.data.root_pos_w.torch[0].detach().cpu().numpy()[None, :]
                sand.update(foot, np.array([[0, 0, 15 * 3.71]]), np.array([0.0]), env.step_dt)
                env.step(zeros)
            after = probe.data.root_pos_w.torch[0, 2].item()
            assert before - after > 0.003, (before, after)
            # CPU scene queries retain stale poses for tensor-moved GPU bodies here.
            # Check actual loaded-foot descent and live physics tensors instead.
            live = sand.columns.data.body_link_pos_w.torch[0, :, 2].cpu().numpy() + COLUMN_DEPTH / 2
            assert np.allclose(live, sand.field.heights.ravel(), atol=1e-5)
            verts = np.asarray(sand.mesh.GetPointsAttr().Get())
            assert np.allclose(verts[:, 2], live, atol=1e-5)
            volume = (sand.field.heights - sand.field.initial).sum() * CELL * CELL
            assert abs(volume) < 1e-10, volume
            env.reset()
            assert np.allclose(sand.field.heights, sand.field.initial)
            print(
                f"LEGACY_HEURISTIC_CHECK not_material_validation foot_drop={before - after:.5f} "
                f"volume_error={volume:.3g} physical_visible_heights_match=True",
                flush=True,
            )
        finally:
            env.close()


if __name__ == "__main__":
    main()
