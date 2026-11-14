"""Headless two-way soil/foot coupling benchmark; NOT Mars material validation.

A dynamic 15 kg plate falls onto a finite, non-cohesive MPM soil bed. Toggle
feedback off as a negative control. No visual particles, prescribed footprints,
support forces after release, camera or policy are involved. An initial force
fixture holds the plate for 0.5 seconds while the bed settles.

Run with /home/eddie/env_isaaclab/bin/python. Logs belong in outputs/log/.
"""

import argparse
import hashlib
import importlib.metadata
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverImplicitMPM, SolverMuJoCo
from newton.solvers.experimental.coupled import SolverCoupledProxy


@wp.kernel
def fixture_force(
    forces: wp.array[wp.spatial_vector], holding: wp.array[int], weight: float, horizontal_load: wp.array[float]
):
    if holding[0] == 1:
        forces[0] = wp.spatial_vector(0.0, 0.0, weight, 0.0, 0.0, 0.0)
    else:
        forces[0] = wp.spatial_vector(horizontal_load[0], 0.0, 0.0, 0.0, 0.0, 0.0)


def main():
    benchmark_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--seconds", type=float, default=1.5)
    parser.add_argument("--gravity", type=float, default=3.71)
    parser.add_argument("--coupling", choices=("two_way", "one_way"), default="two_way")
    parser.add_argument("--friction", type=float, default=0.68)
    parser.add_argument("--density", type=float, default=1400)
    parser.add_argument("--young-modulus", type=float, default=1e7)
    parser.add_argument(
        "--background-regularization",
        type=float,
        default=1.0,
        help="Solver regularization; NOT physical atmospheric drag",
    )
    parser.add_argument(
        "--shear-force", type=float, default=0.0, help="External horizontal force in N, applied 0.75 s after release"
    )
    parser.add_argument("--grid-type", choices=("sparse", "fixed"), default="sparse")
    parser.add_argument("--max-iterations", type=int, default=100)
    args = parser.parse_args()
    if min(args.voxel_size, args.dt, args.seconds, args.gravity, args.density, args.young_modulus) <= 0:
        parser.error("Resolution, time, gravity, density and stiffness must be positive")
    if not all(np.isfinite(value) for value in vars(args).values() if isinstance(value, (float, int))):
        parser.error("Numeric parameters must be finite")
    if args.background_regularization <= 0 or args.friction < 0 or args.shear_force < 0 or args.max_iterations < 1:
        parser.error("Regularization/iterations must be positive; friction/shear must be nonnegative")
    wp.init()
    wp.set_device("cuda:0")
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -args.gravity))
    SolverImplicitMPM.register_custom_attributes(builder)
    mass = 15.0
    dimensions = np.array([0.25, 0.12, 0.04])

    def foot_bottom(pose):
        # Account for tilt: lowest corner of the oriented box, not center minus hz.
        x, y, z, w = pose[3:]
        rotation_z = np.array([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)])
        return float(pose[2] - np.abs(rotation_z) @ (dimensions / 2))

    inertia = (
        mass
        / 12
        * np.array(
            [
                dimensions[1] ** 2 + dimensions[2] ** 2,
                dimensions[0] ** 2 + dimensions[2] ** 2,
                dimensions[0] ** 2 + dimensions[1] ** 2,
            ]
        )
    )
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0, 0, 0.24), wp.quat_identity()),
        mass=mass,
        inertia=wp.mat33(np.diag(inertia)),
        label="test_foot",
    )
    shape_cfg = newton.ModelBuilder.ShapeConfig(density=0, mu=0.6)
    builder.add_shape_box(body, hx=0.125, hy=0.06, hz=0.02, cfg=shape_cfg)
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.6))
    # Cell-center quadrature: total material-point mass is exactly rho * bed volume.
    lower, upper = np.array([-0.4, -0.3, 0.0]), np.array([0.4, 0.3, 0.16])
    counts = np.ceil((upper - lower) / (args.voxel_size / 2)).astype(int)
    spacing = (upper - lower) / counts
    positions = np.stack(
        np.meshgrid(*[lower[i] + (np.arange(counts[i]) + 0.5) * spacing[i] for i in range(3)], indexing="ij"), -1
    ).reshape(-1, 3)
    rng = np.random.default_rng(42)
    positions += rng.uniform(-0.05, 0.05, positions.shape) * spacing
    particle_mass = float(np.prod(spacing) * args.density)
    # Newton MPM uses V = 8*r^3 (a material-point cell), NOT sphere volume.
    # This radius is numerical support, not the radius of an individual grain.
    radius = float(np.cbrt(np.prod(spacing)) / 2)
    assert np.isclose(particle_mass / (8 * radius**3), args.density)
    builder.add_particles(
        pos=positions.tolist(),
        vel=np.zeros_like(positions).tolist(),
        mass=[particle_mass] * len(positions),
        radius=[radius] * len(positions),
        custom_attributes={
            "mpm:friction": [args.friction] * len(positions),
            "mpm:young_modulus": [args.young_modulus] * len(positions),
            "mpm:yield_stress": [0.0] * len(positions),
            "mpm:tensile_yield_ratio": [0.0] * len(positions),
            "mpm:damping": [0.0] * len(positions),
        },
    )
    model = builder.finalize()
    assert np.isclose(model.body_mass.numpy()[0], mass), model.body_mass.numpy()
    expected_soil_mass = float(np.prod(upper - lower) * args.density)
    assert np.isclose(model.particle_mass.numpy().sum(), expected_soil_mass, rtol=1e-6)
    mpm_cfg = SolverImplicitMPM.Config(
        voxel_size=args.voxel_size,
        grid_type=args.grid_type,
        grid_padding=0 if args.grid_type == "sparse" else int(np.ceil(0.4 / args.voxel_size)),
        max_active_cell_count=1 << 18,
        max_iterations=args.max_iterations,
        tolerance=1e-5,
        strain_basis="P0",
        critical_fraction=0.0,
        air_drag=args.background_regularization,
    )
    solver = SolverCoupledProxy(
        model=model,
        entries=[
            SolverCoupledProxy.Entry(
                name="rigid",
                solver=lambda view: SolverMuJoCo(view, use_mujoco_contacts=False, njmax=100),
                bodies=[body],
                joints=list(range(model.joint_count)),
                substeps=2,
            ),
            SolverCoupledProxy.Entry(
                name="soil",
                solver=lambda view: SolverImplicitMPM(view, config=mpm_cfg),
                particles=list(range(model.particle_count)),
                in_place=True,
            ),
        ],
        coupling=SolverCoupledProxy.Config(
            proxies=[
                SolverCoupledProxy.Proxy(
                    source="rigid",
                    destination="soil",
                    bodies=[body],
                    mode="lagged",
                    mass_scale=1.0,
                    proxy_relaxation=1.0 if args.coupling == "two_way" else 0.0,
                    collision_pipeline=lambda _model: None,
                )
            ],
            iterations=1,
        ),
    )
    state = model.state()
    mpm = solver.solver("soil")
    # Check the solver's actual volume convention, not only the input arithmetic.
    represented_density = mpm._mpm_model.particle_density.numpy()
    assert np.allclose(represented_density, args.density, rtol=1e-6)
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    control = model.control()
    collision = newton.CollisionPipeline(model, soft_contact_max=0)
    contacts = collision.contacts()
    holding = wp.ones(1, dtype=int)
    horizontal_load = wp.zeros(1, dtype=float)

    def step():
        state.clear_forces()
        wp.launch(fixture_force, dim=1, inputs=[state.body_f, holding, mass * args.gravity, horizontal_load])
        collision.collide(state, contacts)
        solver.step(state, state, control, contacts, args.dt)

    print(
        f"BENCHMARK_INIT particles={len(positions)} soil_mass_kg={expected_soil_mass} "
        f"foot_mass_kg={mass} coupling={args.coupling}",
        flush=True,
    )
    started = perf_counter()
    with wp.ScopedCapture() as capture:
        step()
    samples = []
    failure = None
    hold_steps = round(0.5 / args.dt)
    impulse_history = []
    for index in range(hold_steps + round(args.seconds / args.dt)):
        if index == hold_steps:
            holding.zero_()
        if index == hold_steps + round(0.75 / args.dt):
            horizontal_load.fill_(args.shear_force)
        wp.capture_launch(capture.graph)
        impulses, _, collider_ids = mpm.collect_collider_impulses(state)
        ids = collider_ids.numpy()
        bodies = mpm.collider_body_index.numpy()
        valid = (ids >= 0) & (ids < len(bodies))
        on_body = np.zeros(len(ids), dtype=bool)
        on_body[valid] = bodies[ids[valid]] >= 0
        impulse = impulses.numpy()[on_body].sum(axis=0)
        impulse_history.append(impulse)
        soil_force = impulse / args.dt
        if index % max(1, round(0.05 / args.dt)) == 0:
            q, v = state.body_q.numpy()[0], state.body_qd.numpy()[0]
            points = state.particle_q.numpy()
            if not all(np.isfinite(a).all() for a in (q, v, points, state.particle_qd.numpy())):
                failure = f"Nonfinite state at step {index + 1}"
                break
            samples.append(
                dict(
                    time_s=(index + 1 - hold_steps) * args.dt,
                    foot_bottom_m=foot_bottom(q),
                    vz_m_s=float(v[2]),
                    foot_x_m=float(q[0]),
                    vx_m_s=float(v[0]),
                    soil_force_z_n=float(soil_force[2]),
                    soil_force_x_n=float(soil_force[0]),
                )
            )
            if index % max(1, round(0.25 / args.dt)) == 0:
                print("BENCHMARK_PROGRESS " + json.dumps(samples[-1]), flush=True)
    wp.synchronize()
    final_points = state.particle_q.numpy()
    final_pose = state.body_q.numpy()[0]
    if not all(
        np.isfinite(a).all() for a in (final_points, final_pose, state.particle_qd.numpy(), state.body_qd.numpy())
    ):
        failure = failure or "Nonfinite final state"
    tail = np.array(impulse_history[-round(0.25 / args.dt) :])
    report = dict(
        status="solver_verification_only_not_Mars_validation",
        parameters=vars(args),
        benchmark_sha256=benchmark_hash,
        material_note="Provisional non-cohesive end-member; internal friction and stiffness NOT calibrated.",
        regularization_note="Background regularization is numerical, not a Mars atmosphere model.",
        density_reference="https://ntrs.nasa.gov/citations/20200003046",
        software={key: importlib.metadata.version(key) for key in ("newton", "warp-lang", "mujoco-warp")},
        model=dict(
            particles=model.particle_count,
            soil_mass_kg=expected_soil_mass,
            foot_mass_kg=mass,
            bed_lower_m=lower.tolist(),
            bed_upper_m=upper.tolist(),
            seed=42,
            point_volume_m3=8 * radius**3,
            represented_density_kg_m3=float(represented_density.mean()),
            rigid_substeps=2,
            mpm_max_active_cells=1 << 18,
            mpm_tolerance=1e-5,
            coupling_mode="lagged",
            proxy_mass_scale=1.0,
            interface_friction=0.6,
            material={
                name: float(getattr(model.mpm, name).numpy()[0])
                for name in (
                    "young_modulus",
                    "poisson_ratio",
                    "friction",
                    "yield_pressure",
                    "tensile_yield_ratio",
                    "yield_stress",
                    "hardening",
                    "dilatancy",
                    "viscosity",
                    "damping",
                )
            },
        ),
        final_foot_bottom_m=foot_bottom(final_pose),
        final_foot_x_m=float(final_pose[0]),
        last_quarter_second_mean_soil_force_n=(tail.mean(axis=0) / args.dt).tolist(),
        expected_foot_weight_n=mass * args.gravity,
        particle_min_z_m=float(final_points[:, 2].min()),
        particle_rms_displacement_m=float(np.sqrt(np.mean(np.sum((final_points - positions) ** 2, axis=1)))),
        finite=failure is None,
        failure=failure,
        wall_seconds=perf_counter() - started,
        samples=samples,
    )
    folder = Path("outputs/physics")
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(ZoneInfo("America/Vancouver")).strftime("%Y%m%d-%H%M%S-%f")
    output = folder / f"mpm-{args.coupling}-{stamp}.json"

    def json_safe(value):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        return value

    report = json_safe(report)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("BENCHMARK_RESULT " + json.dumps({k: v for k, v in report.items() if k != "samples"}), flush=True)
    print(f"BENCHMARK_SAVED {output.resolve()}", flush=True)
    if not report["finite"]:
        raise RuntimeError("MPM benchmark diverged")


if __name__ == "__main__":
    main()
