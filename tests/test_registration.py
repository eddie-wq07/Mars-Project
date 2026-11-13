"""Check the standalone project's Gymnasium registration."""

import gymnasium as gym

import mars_project.tasks.h1  # noqa: F401


def test_task_registrations():
    spec = gym.spec("Mars-Project-H1")
    assert spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv"
    assert spec.kwargs["env_cfg_entry_point"] == "mars_project.tasks.h1:MarsH1EnvCfg"
    assert spec.kwargs["rsl_rl_cfg_entry_point"] == "mars_project.tasks.h1:MarsH1PPORunnerCfg"
