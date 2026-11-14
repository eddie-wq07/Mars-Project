"""Physical limits and conservation, independent of rendering or simulator startup."""

import numpy as np
import pytest

from mars_project.environment.airborne_dust import Atmosphere, DustParcels, advance_dust
from mars_project.environment.dust_volume import extinction_grid
from mars_project.environment.soil_surface import refine_terrain


def test_slip_and_settling_have_correct_size_and_pressure_dependence():
    mars = Atmosphere()
    sizes = np.array([1e-6, 2e-6, 5e-6, 10e-6])
    tau = mars.relaxation_time(sizes)
    assert 0.01 < mars.density < 0.02
    assert 4e-6 < mars.mean_free_path < 5e-6
    assert Atmosphere(pressure=1220.0).mean_free_path == pytest.approx(mars.mean_free_path / 2)
    assert np.all(np.diff(tau) > 0)
    continuum = 2 * mars.grain_density * sizes**2 / (9 * mars.viscosity)
    assert np.all(tau > continuum)
    p, v = advance_dust(np.zeros((4, 3)), np.zeros((4, 3)), sizes, 1.0, mars)
    terminal = mars.gravity * tau * (1 - mars.density / mars.grain_density)
    np.testing.assert_allclose(v[:, 2], terminal, atol=1e-6)
    assert np.all(p[:, 2] < 0)


def test_exact_integrator_is_independent_of_step_partition():
    atmosphere = Atmosphere(wind=(0.3, -0.1, 0.0))
    p, v, r = np.array([[0.0, 0.0, 1.0]]), np.array([[0.5, 0.2, 1.0]]), np.array([2e-6])
    expected_p, expected_v = advance_dust(p, v, r, 0.1, atmosphere)
    for _ in range(100):
        p, v = advance_dust(p, v, r, 0.001, atmosphere)
    np.testing.assert_allclose(p, expected_p, atol=1e-12)
    np.testing.assert_allclose(v, expected_v, atol=1e-12)
    with pytest.raises(ValueError, match="Reynolds"):
        advance_dust(p, v * 1e9 + 1e6, r, 0.1, atmosphere)


def test_emission_and_deposition_preserve_mass_and_account_for_momentum():
    cloud = DustParcels(capacity=4)
    mass = np.array([1e-4, 2e-4])
    velocity = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])
    accepted = cloud.emit(np.array([[0.0, 0.0, 0.01], [0.0, 0.0, 0.02]]), velocity, mass, np.array([7, 8]))
    np.testing.assert_array_equal(accepted, [7])
    assert cloud.mass.sum() == pytest.approx(mass[0])
    momentum = mass[0] * velocity[0]
    np.testing.assert_allclose(np.sum(cloud.mass[:, None] * cloud.velocity, axis=0), momentum)
    for _ in range(300):
        cloud.step(0.02, lambda xy: np.zeros(len(xy)))
    assert cloud.mass.sum() + cloud.deposited == pytest.approx(cloud.emitted)
    final = np.sum(cloud.mass[:, None] * cloud.velocity, axis=0) + cloud.deposit_impulse
    np.testing.assert_allclose(final, momentum + cloud.drag_impulse + cloud.gravity_impulse, atol=1e-12)


def test_render_reconstruction_preserves_optical_mass():
    positions = np.array([[0.0, 0.0, 0.1], [0.05, 0.01, 0.12]])
    areas = np.array([0.002, 0.0001])
    field, _ = extinction_grid(positions, areas)
    assert np.all(field >= 0)
    assert field.sum() * 0.025**3 == pytest.approx(areas.sum(), rel=1e-5)


def test_terrain_replacement_keeps_source_height_uv_and_shared_edges():
    x, y = np.meshgrid(np.arange(-2.0, 3.0), np.arange(-2.0, 3.0))
    points = np.stack((x.ravel(), y.ravel(), (0.1 * x + 0.2 * y).ravel()), -1)
    ids = np.arange(25).reshape(5, 5)
    faces = np.concatenate(
        (
            np.stack((ids[:-1, :-1], ids[:-1, 1:], ids[1:, :-1]), -1).reshape(-1, 3),
            np.stack((ids[:-1, 1:], ids[1:, 1:], ids[1:, :-1]), -1).reshape(-1, 3),
        )
    )
    uv = points[:, :2] / 4 + 0.5
    remaining, replacement, faces_new, uv_new, _ = refine_terrain(
        points, faces, uv, (-0.9, -0.9, 0.9, 0.9), spacing=0.1
    )
    np.testing.assert_allclose(replacement[:, 2], 0.1 * replacement[:, 0] + 0.2 * replacement[:, 1], atol=1e-7)
    np.testing.assert_allclose(uv_new, replacement[:, :2] / 4 + 0.5, atol=1e-7)
    assert len(remaining) < len(faces)
    assert faces_new.max() < len(replacement)
    assert replacement[:, 0].min() == -1 and replacement[:, 0].max() == 1
