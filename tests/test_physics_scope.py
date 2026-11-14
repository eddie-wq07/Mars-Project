"""MPM playback must load a coupled material and retain an explicit rigid baseline."""

import copy
import sys

import numpy as np
import pytest
import torch

from mars_project.environment.mpm_soil import (
    DENSITY,
    POLICY_JOINT_NAMES,
    THICKNESS,
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
    create_soil_height_scanner,
    mpm_height_scan,
    particle_surface_heights,
    soil_particles,
)
from mars_project.scene_runner import parse_args
from mars_project.scenes.canyon import configure_canyon_scene
from mars_project.tasks.h1 import MarsH1EnvCfg


def test_canyon_defaults_to_coupled_mpm_without_changing_motor_gains():
    cfg = MarsH1EnvCfg()
    cfg.scene.num_envs = 1
    actuators = copy.deepcopy(cfg.scene.robot.actuators)
    configure_canyon_scene(cfg)
    assert cfg.scene.terrain is None
    assert cfg.scene.soil_bed.spawn.usd_path.endswith("local_ground.usdc")
    assert len(cfg.scene.soil.spawn.positions) == 520000
    assert cfg.scene.robot.init_state.pos[2] == pytest.approx(1.05)
    assert cfg.scene.contact_forces is None
    assert cfg.terminations.soil_fall is not None
    assert cfg.actions.joint_pos.preserve_order
    assert tuple(cfg.actions.joint_pos.joint_names) == POLICY_JOINT_NAMES
    for name in ("joint_pos", "joint_vel"):
        selection = getattr(cfg.observations.policy, name).params["asset_cfg"]
        assert selection.preserve_order
        assert tuple(selection.joint_names) == POLICY_JOINT_NAMES
    assert getattr(cfg.scene, "regolith", None) is None
    assert getattr(cfg.events, "deform_regolith", None) is None
    assert cfg.observations.policy.height_scan.func is mpm_height_scan
    assert cfg.scene.height_scanner.class_type is create_soil_height_scanner
    assert cfg.scene.height_scanner.mesh_prim_paths == ["/World/envs/env_0/SoilBed"]
    coupling = cfg.sim.physics.solver_cfg
    assert coupling.proxies[0].proxy_relaxation == 1.0
    assert coupling.entries[1].all_particles
    assert coupling.entries[1].in_place
    for name, actuator in cfg.scene.robot.actuators.items():
        assert actuator.damping == actuators[name].damping


def test_rigid_baseline_keeps_original_sensor_and_backend():
    cfg = MarsH1EnvCfg()
    cfg.scene.num_envs = 1
    height_sensor = cfg.observations.policy.height_scan.func
    physics = cfg.sim.physics
    configure_canyon_scene(cfg, soil="rigid")
    assert cfg.scene.terrain.usd_path.endswith("local_ground.usdc")
    assert getattr(cfg.scene, "soil", None) is None
    assert cfg.observations.policy.height_scan.func is height_sensor
    assert cfg.sim.physics is physics
    assert cfg.scene.contact_forces is not None


def test_particle_mass_density_and_terrain_coverage():
    from mars_project.environment.regolith import terrain_height

    points, mass, radius = soil_particles()
    assert mass / (8 * radius**3) == pytest.approx(DENSITY)
    assert len(points) * mass == pytest.approx((X_MAX - X_MIN) * (Y_MAX - Y_MIN) * THICKNESS * DENSITY)
    height = points[:, 2] - terrain_height(points[:, 0], points[:, 1])
    assert np.isfinite(points).all()
    assert height.min() > -THICKNESS
    assert height.max() < 0


def test_height_observation_follows_moving_particles_and_falls_back_to_bed():
    points = torch.tensor([[0.0, 0.0, 0.12], [0.0, 0.0, 0.08], [0.5, 0.5, 0.16]])
    queries = torch.tensor([[0.0, 0.0], [0.5, 0.5], [2.0, 1.0], [-3.0, 0.0]])
    bed = torch.tensor([0.0, 0.0, 0.03, 0.04])
    initial = particle_surface_heights(points, queries, bed, radius=0.01)
    assert torch.allclose(initial, torch.tensor([0.13, 0.17, 0.03, 0.04]))
    points[0, 2] -= 0.03
    deformed = particle_surface_heights(points, queries, bed, radius=0.01)
    assert deformed[0].item() == pytest.approx(0.10)


def test_soil_cli_defaults_and_explicit_rigid_baseline(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["view_scene.py", "--appearance", "grand-canyon"])
    assert parse_args().soil == "mpm"
    monkeypatch.setattr(sys, "argv", ["view_scene.py", "--appearance", "grand-canyon", "--soil", "rigid"])
    assert parse_args().soil == "rigid"
    monkeypatch.setattr(sys, "argv", ["view_scene.py", "--appearance", "earth"])
    assert parse_args().soil == "rigid"


def test_default_motor_damping_does_not_depend_on_planet(monkeypatch):
    for planet in ("earth", "mars"):
        monkeypatch.setattr(sys, "argv", ["view_scene.py", "--gravity", planet])
        assert parse_args().joint_resistance_scale == 0.0
