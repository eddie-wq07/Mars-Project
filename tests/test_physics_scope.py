"""Playback must not silently substitute prescribed deformation for soil mechanics."""

import copy
import sys

from mars_project.scene_runner import parse_args
from mars_project.scenes.canyon import configure_canyon_scene
from mars_project.tasks.h1 import MarsH1EnvCfg


def test_canyon_uses_rigid_ground_without_prescribed_soil_events():
    cfg = MarsH1EnvCfg()
    cfg.scene.num_envs = 1
    height_sensor = cfg.observations.policy.height_scan.func
    actuators = copy.deepcopy(cfg.scene.robot.actuators)
    configure_canyon_scene(cfg)
    assert cfg.scene.terrain.usd_path.endswith("local_ground.usdc")
    assert getattr(cfg.scene, "regolith", None) is None
    assert getattr(cfg.events, "deform_regolith", None) is None
    assert cfg.observations.policy.height_scan.func is height_sensor
    for name, actuator in cfg.scene.robot.actuators.items():
        assert actuator.damping == actuators[name].damping


def test_default_motor_damping_does_not_depend_on_planet(monkeypatch):
    for planet in ("earth", "mars"):
        monkeypatch.setattr(sys, "argv", ["view_scene.py", "--gravity", planet])
        assert parse_args().joint_resistance_scale == 0.0
