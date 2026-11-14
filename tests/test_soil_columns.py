"""Physical invariants of the shallow, depth-averaged terrain approximation."""

import pytest
import torch

from mars_project.environment.soil_columns import column_depth, surface_fraction


def project(points, volume):
    return column_depth(points, volume, 40, 30, 0.0, 0.0, 0.04)


def test_vertical_jitter_does_not_raise_ground_and_horizontal_flow_leaves_a_rut():
    x, y, z = torch.meshgrid(
        torch.arange(0.01, 1.6, 0.02), torch.arange(0.01, 1.2, 0.02), torch.tensor([-0.03, -0.01]), indexing="ij"
    )
    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), -1)
    volumes = torch.full((len(points),), 0.02**3)
    initial = project(points, volumes)
    points[:, 2] += 0.01 * torch.sin(points[:, 0] * 50)
    assert torch.equal(initial, project(points, volumes))
    region = (points[:, 0] > 0.7) & (points[:, 0] < 0.8) & (points[:, 1] > 0.5) & (points[:, 1] < 0.7)
    points[region, 0] -= 0.12
    moved = project(points, volumes)
    assert float((moved - initial).min()) < -0.01
    assert float((moved - initial).max()) > 0.01
    assert float(moved.sum()) == pytest.approx(float(initial.sum()), rel=1e-6)
    assert torch.equal(moved, project(points, volumes)), "Unloaded static material should retain the rut"


def test_subcell_translation_is_continuous_and_conserves_interior_volume():
    volume = torch.tensor([0.02**3])
    previous = None
    for x in torch.linspace(0.77, 0.81, 41):
        points = torch.tensor([[x, 0.6, -0.01]])
        depth = project(points, volume)
        assert float(depth.sum()) * 0.04**2 == pytest.approx(float(volume[0]), rel=1e-6)
        if previous is not None:
            assert float((depth - previous).abs().max()) < 0.0001
        previous = depth


def test_airborne_material_stays_excluded_at_apex_and_rejoins_continuously():
    height = torch.tensor([0.08, 0.10, 0.12, 0.10, 0.08])
    assert torch.count_nonzero(surface_fraction(height, torch.zeros_like(height), 0.01, 0.04)) == 0
    descending = surface_fraction(torch.linspace(0.07, 0.03, 41), torch.tensor(0.0), 0.01, 0.04)
    assert torch.all(descending[1:] >= descending[:-1])
    assert float((descending[1:] - descending[:-1]).max()) < 0.026
    assert float(descending[-1]) == pytest.approx(1.0)
