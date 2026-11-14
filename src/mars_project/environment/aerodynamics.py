"""Prescribed-air, per-link quadratic drag; no changes to actuator damping."""

import torch

from isaaclab.utils.math import quat_apply_inverse

from mars_project.environment.airborne_dust import Atmosphere


def air_density(profile):
    """Use the same ideal-gas reference states as the existing aerosol model."""
    if profile == "off":
        return 0.0
    if profile == "mars":
        return Atmosphere().density
    if profile == "earth":
        return Atmosphere(pressure=101325.0, temperature=288.15, molar_mass=0.02897).density
    raise ValueError(f"Unknown air-drag profile: {profile}")


def box_drag_force(velocity_w, orientation_xyzw, dimensions, density, coefficient=1.0, wind_w=None):
    """Return world forces at link CoMs using each box's projected frontal area.

    Shapes broadcast over environments/links. Link dimensions are in metres in
    the link frame, velocity/wind in m/s, density in kg/m³, output in newtons.
    Cd is a scenario assumption. No lift, wake shielding or spin torque is solved.
    """
    relative_w = velocity_w if wind_w is None else velocity_w - wind_w
    relative_b = quat_apply_inverse(orientation_xyzw, relative_w)
    # A * speed = |vx| dy dz + |vy| dx dz + |vz| dx dy.
    # This equivalent form avoids normalizing a zero-speed vector.
    face_areas = dimensions[..., [1, 0, 0]] * dimensions[..., [2, 2, 1]]
    area_times_speed = (relative_b.abs() * face_areas).sum(dim=-1, keepdim=True)
    return -0.5 * density * coefficient * area_times_speed * relative_w


def link_box_dimensions(stage, robot_prim_path, body_names):
    """Bound each link's visuals in its own frame, matching names, not USD order.

    H1 collision geometry only covers feet/torso, so it cannot represent exposed
    arms and legs. Visual bounds intentionally overestimate complex silhouettes.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    from isaaclab.sim.utils import find_first_matching_prim

    root = find_first_matching_prim(robot_prim_path, stage=stage)
    if root is None:
        raise ValueError(f"Robot prim not found: {robot_prim_path}")
    bodies = {
        prim.GetName(): prim
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    }
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
    dimensions = []
    for name in body_names:
        body = bodies[name]
        visuals = body.GetChild("visuals")
        if not visuals.IsValid():
            raise ValueError(f"No visual geometry for drag area: {name}")
        bounds = cache.ComputeRelativeBound(visuals, body).ComputeAlignedRange()
        size = tuple(bounds.GetSize())
        if bounds.IsEmpty() or not all(0 < value < float("inf") for value in size):
            raise ValueError(f"Invalid drag dimensions for {name}: {size}")
        dimensions.append(size)
    return torch.tensor(dimensions, dtype=torch.float32)
