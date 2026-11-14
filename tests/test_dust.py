"""Fine particles emit only during loaded foot motion or touchdown."""

import numpy as np

from mars_project.environment.dust import DustCloud, drag_velocity
from mars_project.environment.resistance import EARTH, MARS


def test_fine_particle_scale_and_contact_gating():
    feet = np.array([[0.0, 0.0, 0.05]])
    velocity = np.array([[0.2, 0, 0]])
    force = np.array([[0.0, 0.0, 30.0]])
    cloud = DustCloud(MARS)
    assert np.all((cloud.radii >= 0.000075) & (cloud.radii <= 0.000175))
    for _ in range(50):
        cloud.update(0.02, feet, velocity, force * 0)
    assert cloud.total_emitted == 0
    for _ in range(50):
        cloud.update(0.02, feet, velocity, force)
    assert 0 < cloud.total_emitted <= 600
    before = cloud.total_emitted
    for _ in range(100):
        cloud.update(0.02, feet, velocity * 0, force)
    assert cloud.total_emitted == before
    assert not (cloud.life > 0).any()
    cloud.clear()
    assert cloud.was_loaded is None
    before = cloud.total_emitted
    cloud.update(0.02, feet + np.array([[10, 0, 0]]), velocity, force)
    assert cloud.total_emitted == before


def test_mars_particle_drag_is_lower_without_using_joint_gains():
    v = np.array([[1.0, 0, 0]])
    earth = drag_velocity(v, 0.1, EARTH.air_density)
    mars = drag_velocity(v, 0.1, MARS.air_density)
    assert 0 < earth[0, 0] < mars[0, 0] < 1
    assert np.isfinite(drag_velocity(v, 100, EARTH.air_density)).all()
