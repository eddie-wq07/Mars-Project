"""Physical invariants of the simple robot drag approximation."""

import copy
import math
import sys

import pytest
import torch

from mars_project.environment.aerodynamics import air_density, box_drag_force
from mars_project.environment.air_drag import configure_air_drag
from mars_project.scene_runner import parse_args
from mars_project.tasks.h1 import MarsH1EnvCfg


def drag(velocity, *, dimensions=(2.0, 3.0, 4.0), rotation=(0.0, 0.0, 0.0, 1.0), profile="earth", wind=None):
    return box_drag_force(
        torch.tensor(velocity),
        torch.tensor(rotation),
        torch.tensor(dimensions),
        air_density(profile),
        wind_w=None if wind is None else torch.tensor(wind),
    )


def test_known_area_force_and_quadratic_speed():
    force = drag((1.0, 0.0, 0.0))
    assert force.tolist() == pytest.approx([-0.5 * air_density("earth") * 12, 0, 0])
    assert torch.allclose(drag((2.0, 0.0, 0.0)), force * 4)
    assert torch.allclose(drag((-1.0, 0.0, 0.0)), -force)


def test_area_rotates_with_link_and_rotating_everything_rotates_force():
    q = (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
    assert drag((1.0, 0.0, 0.0), rotation=q)[0].item() == pytest.approx(-0.5 * air_density("earth") * 8)
    assert drag((0.0, 1.0, 0.0), rotation=q)[1].item() == pytest.approx(drag((1.0, 0.0, 0.0))[0].item())


def test_density_scaling_still_air_wind_and_disabled_drag():
    assert air_density("mars") / air_density("earth") == pytest.approx(0.0125497, rel=1e-4)
    earth = drag((1.0, 2.0, 3.0))
    mars = drag((1.0, 2.0, 3.0), profile="mars")
    assert torch.allclose(mars, earth * air_density("mars") / air_density("earth"))
    for force in (
        drag((0.0, 0.0, 0.0)),
        drag((1.0, 2.0, 3.0), wind=(1.0, 2.0, 3.0)),
        drag((1.0, 2.0, 3.0), profile="off"),
    ):
        assert torch.isfinite(force).all() and torch.count_nonzero(force) == 0
    assert drag((0.0, 0.0, 0.0), wind=(1.0, 0.0, 0.0))[0] > 0


def test_drag_dissipates_relative_motion_for_batched_links():
    generator = torch.Generator().manual_seed(42)
    v = torch.randn((3, 20, 3), generator=generator)
    q = torch.randn((3, 20, 4), generator=generator)
    q /= torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    sizes = torch.rand((20, 3), generator=generator) + 0.01
    wind = torch.tensor([0.3, -0.2, 0.0])
    force = box_drag_force(v, q, sizes, air_density("earth"), wind_w=wind)
    assert force.shape == v.shape
    assert ((force * (v - wind)).sum(-1) <= 0).all()


def test_configuration_keeps_actuators_and_policy_controls():
    cfg = MarsH1EnvCfg()
    actuators = copy.deepcopy(cfg.scene.robot.actuators)
    joint_action = copy.deepcopy(cfg.actions.joint_pos)
    for gravity, expected in [(-9.81, "earth"), (-3.71, "mars")]:
        cfg.sim.gravity = (0.0, 0.0, gravity)
        configure_air_drag(cfg)
        assert cfg.actions.air_drag.profile == expected
        assert cfg.actions.joint_pos.to_dict() == joint_action.to_dict()
        for name, before in actuators.items():
            assert cfg.scene.robot.actuators[name].to_dict() == before.to_dict()
    configure_air_drag(cfg, "earth")
    assert cfg.actions.air_drag.profile == "earth"
    with pytest.raises(ValueError, match="Unknown"):
        configure_air_drag(cfg, "invalid")


def test_cli_drag_override(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["view_scene.py"])
    assert parse_args().air_drag == "auto"
    monkeypatch.setattr(sys, "argv", ["view_scene.py", "--air-drag", "off"])
    assert parse_args().air_drag == "off"
