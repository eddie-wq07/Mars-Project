"""Added damping must preserve original gains and follow the selected physics profile."""

import copy

import pytest

from mars_project.environment.resistance import EARTH, MARS, apply_joint_resistance, environment_profile
from mars_project.tasks.h1 import MarsH1EnvCfg


def test_baseline_and_profile_damping():
    original = MarsH1EnvCfg()
    baseline = copy.deepcopy(original.scene.robot.actuators)
    for profile, scale in [(EARTH, 0), (EARTH, 1), (MARS, 1)]:
        cfg = copy.deepcopy(original)
        apply_joint_resistance(cfg, profile, scale)
        for name, actuator in cfg.scene.robot.actuators.items():
            before = baseline[name]
            assert actuator.stiffness == before.stiffness
            assert actuator.effort_limit_sim == before.effort_limit_sim
            for key, value in before.damping.items():
                assert actuator.damping[key] == pytest.approx(value + profile.added_joint_damping * scale)
        with pytest.raises(ValueError, match="already applied"):
            apply_joint_resistance(cfg, profile, scale)
    assert 0 < MARS.added_joint_damping < EARTH.added_joint_damping
    assert environment_profile("mars") is MARS


@pytest.mark.parametrize("scale", [-1, float("nan"), float("inf")])
def test_rejects_invalid_damping(scale):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        apply_joint_resistance(MarsH1EnvCfg(), EARTH, scale)
