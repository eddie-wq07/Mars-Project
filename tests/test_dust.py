"""Dust must remain local and contact-driven, with independent atmospheric drag."""

import numpy as np

from mars_project.environment.dust import DustCloud, disturbed_grains, drag_velocity
from mars_project.environment.resistance import EARTH, MARS


def test_emission_requires_loaded_foot_and_moving_grains():
    p = np.array([[0.0, 0.0, 0.03], [2.0, 0.0, 0.03]])
    v = np.array([[0.2, 0, 0], [0.2, 0, 0]])
    feet = np.array([[0.0, 0.0, 0.1]])
    force = np.array([[0.0, 0.0, 30.0]])
    assert disturbed_grains(p, v, feet, force).tolist() == [True, False]
    assert not disturbed_grains(p, v, feet, force * 0).any()
    assert not disturbed_grains(p, v * 0, feet, force).any()
    cloud = DustCloud(MARS)
    for _ in range(50):
        cloud.update(0.02, p, v, feet, force * 0)
    assert cloud.total_emitted == 0
    for _ in range(50):
        cloud.update(0.02, p, v, feet, force)
    assert 0 < cloud.total_emitted <= 160
    assert (cloud.life > 0).sum() <= 512
    for _ in range(100):
        cloud.update(0.02, p, v * 0, feet, force)
    assert not (cloud.life > 0).any()
    cloud.clear()
    assert not cloud.life.any()


def test_mars_particle_drag_is_lower_without_using_joint_gains():
    velocity = np.array([[1.0, 0, 0]])
    earth = drag_velocity(velocity, 0.1, EARTH.air_density)
    mars = drag_velocity(velocity, 0.1, MARS.air_density)
    assert 0 < earth[0, 0] < mars[0, 0] < 1
    assert np.isfinite(drag_velocity(velocity, 100, EARTH.air_density)).all()
