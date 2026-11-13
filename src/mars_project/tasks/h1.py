"""H1 flat-ground locomotion, inheriting NVIDIA's scene and MDP terms.

Robot, observations, actions, rewards, resets and terminations come from
H1FlatEnvCfg. The runner selects gravity and applies scene/preview settings separately.
"""

import gymnasium as gym

from isaaclab.utils import configclass

from isaaclab_tasks.core.velocity.config.h1.agents.rsl_rl_ppo_cfg import H1FlatPPORunnerCfg
from isaaclab_tasks.core.velocity.config.h1.flat_env_cfg import H1FlatEnvCfg


@configclass
class MarsH1EnvCfg(H1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.sim.gravity = (0.0, 0.0, -3.71)
        self.sim.physics.default = self.sim.physics.isaacsim_physx


@configclass
class MarsH1PPORunnerCfg(H1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "mars_h1"


gym.register(
    id="Mars-Project-H1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "mars_project.tasks.h1:MarsH1EnvCfg",
        "rsl_rl_cfg_entry_point": "mars_project.tasks.h1:MarsH1PPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
