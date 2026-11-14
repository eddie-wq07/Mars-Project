"""Depth-averaged surface for a shallow, approximately constant-density sand layer.

Height follows projected material volume, not a column's highest point. Vertical
elastic jitter therefore cannot lift the terrain sheet. Horizontal redistribution
leaves ruts and piles. This approximation omits resolved elastic volume changes,
overhangs and airborne bulk-clump geometry; it does not replace MPM contact forces.
"""

import torch


def quadratic_weight(distance):
    distance = distance.abs()
    return torch.where(
        distance < 0.5,
        0.75 - distance.square(),
        0.5 * (1.5 - distance).clamp(min=0).square(),
    )


def column_depth(points, volumes, nx, ny, xmin, ymin, spacing):
    """Deposit volume/area continuously with a 3x3 quadratic B-spline kernel.

    Interior deposition conserves volume. The fixed boundary guard is wider than
    the kernel support, so clipping at the outer grid cannot affect walking terrain.
    """
    xy = (points[:, :2] - points.new_tensor((xmin, ymin))) / spacing - 0.5
    lower = (xy - 0.5).floor().long()
    depth = points.new_zeros(nx * ny)
    for a in range(3):
        x = lower[:, 0] + a
        wx = quadratic_weight(xy[:, 0] - x)
        for b in range(3):
            y = lower[:, 1] + b
            valid = (x >= 0) & (x < nx) & (y >= 0) & (y < ny)
            weight = wx * quadratic_weight(xy[:, 1] - y)
            depth.scatter_add_(0, y[valid] * nx + x[valid], (volumes * weight / spacing**2)[valid])
    return depth.reshape(ny, nx)


def surface_fraction(point_z, ground_z, radius, spacing):
    """Fade detached bulk material geometrically, independently of velocity sign.

    A half-cell contact band tolerates the coarse MPM interface. Above that band,
    one cell smoothly removes airborne volume; falling material rejoins gradually.
    """
    gap = point_z - radius - ground_z - 0.5 * spacing
    return (1.0 - gap / spacing).clamp(0.0, 1.0)
