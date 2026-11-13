"""Check the standalone project's Gymnasium registration."""

import gymnasium as gym

import mars_project.tasks.h1  # noqa: F401


def test_task_registrations():
    spec = gym.spec("Mars-Project-H1")
    assert spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv"
    assert spec.kwargs["env_cfg_entry_point"] == "mars_project.tasks.h1:MarsH1EnvCfg"
    assert spec.kwargs["rsl_rl_cfg_entry_point"] == "mars_project.tasks.h1:MarsH1PPORunnerCfg"


def test_earth_spawn_faces_outward_without_random_initial_motion():
    import math

    from mars_project.scenes.setup import configure_preview
    from mars_project.tasks.h1 import MarsH1EnvCfg

    cfg = MarsH1EnvCfg()
    configure_preview(cfg, "earth")
    assert cfg.scene.robot.init_state.pos == (0.0, 0.0, 1.05)
    # Installed Isaac Lab uses xyzw; wxyz identity would flip the robot upside down.
    assert cfg.scene.robot.init_state.rot == (0.0, 0.0, 0.0, 1.0)
    pose = cfg.events.reset_base.params["pose_range"]
    assert set(pose) == {"yaw"}
    assert pose["yaw"][0] == pose["yaw"][1]
    assert math.sin(pose["yaw"][0]) < -0.99  # outward trail is toward world -Y
    assert cfg.events.reset_base.params["velocity_range"] == {}
    assert cfg.scene.robot.init_state.lin_vel == (0.0, 0.0, 0.0)
    assert cfg.scene.robot.init_state.ang_vel == (0.0, 0.0, 0.0)
    assert cfg.events.add_base_mass is None
    assert cfg.seed == 42
