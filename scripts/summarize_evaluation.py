"""Independently audit every saved trajectory, then write comparison JSON and plots."""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from mars_project.evaluation import Protocol, classify_sample, route_rotation, summarize


def merge_mars_shards(root, shards):
    """Preserve worker artifacts and assemble a fully audited, non-overlapping batch."""
    expected = json.loads((root / "earth" / "manifest.json").read_text())
    batches = [audit_condition(shard, "mars") for shard in shards]
    manifests = [batch[0] for batch in batches]
    combined = sorted((r for _, rows, _ in batches for r in rows), key=lambda r: r["seed"])
    assert [r["seed"] for r in combined] == expected["seeds"], "Missing or duplicated shard seeds"
    first = manifests[0]
    for manifest in manifests[1:]:
        for key in first.keys() - {"requested_trials", "seeds"}:
            assert first[key] == manifest[key], f"Shards disagree on {key}"
    target = root / "mars"
    target.mkdir(exist_ok=False)
    for shard in shards:
        for path in (shard / "mars").glob("trial-*.*"):
            with path.open("rb") as source, (target / path.name).open("xb") as destination:
                shutil.copyfileobj(source, destination)
    manifest = dict(
        first,
        requested_trials=len(combined),
        seeds=expected["seeds"],
        execution_shards=[dict(path=str(shard), seeds=m["seeds"]) for shard, m in zip(shards, manifests, strict=True)],
    )
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (target / "summary.json").write_text(json.dumps(dict(summarize(combined), complete=True), indent=2) + "\n")


def audit_condition(root, condition):
    folder = root / condition
    assert not (folder / "invalid-run.json").exists(), f"Rejected diagnostic run: {folder}"
    manifest = json.loads((folder / "manifest.json").read_text())
    protocol = Protocol(**manifest["protocol"])
    results, trajectories = [], []
    files = sorted(folder.glob("trial-*.json"))
    assert len(files) == manifest["requested_trials"], f"Incomplete {condition} batch"
    assert len(list(folder.glob("trial-*.npz"))) == len(files), "Missing/orphaned trajectory"
    rotation = route_rotation(manifest["route_yaw_rad"])
    for path in files:
        result = json.loads(path.read_text())
        assert result["condition"] == condition
        with np.load(folder / result["trajectory_file"], allow_pickle=False) as loaded:
            trace = {key: loaded[key] for key in loaded.files}
        count = result["samples"]
        assert all(len(value) == count and np.isfinite(value).all() for value in trace.values())
        assert np.allclose(trace["time_s"], np.arange(count) * manifest["control_dt"], atol=1e-8)
        assert trace["time_s"][-1] == result["duration_s"] <= protocol.timeout_s
        assert np.allclose(trace["root_position_w"][0], result["start_position_w"])
        assert np.allclose(trace["root_quaternion_xyzw"][0], result["initial_quaternion_xyzw"])
        assert np.allclose(trace["root_velocity_w"][0], result["initial_velocity_w"])
        yaw = manifest["route_yaw_rad"] + result["perturbation"]["yaw_rad"]
        expected_quaternion = np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])
        actual_quaternion = trace["root_quaternion_xyzw"][0]
        assert (
            min(
                np.linalg.norm(actual_quaternion - expected_quaternion),
                np.linalg.norm(actual_quaternion + expected_quaternion),
            )
            < 1e-5
        )
        assert abs(trace["root_position_w"][0, 2] - 1.05) < 1e-5
        assert np.allclose(trace["root_velocity_w"][0, 2:5], 0.0, atol=1e-5)
        assert abs(trace["root_velocity_w"][0, 5] - result["perturbation"]["yaw_rate_rad_s"]) < 1e-5
        assert np.allclose(trace["root_position_w"][-1], result["final_position_w"])
        assert (np.linalg.norm(np.diff(trace["root_position_w"], axis=0), axis=1) < 0.25).all(), "Discontinuous state"
        route = (trace["root_position_w"][:, :2] - trace["root_position_w"][0, :2]) @ rotation
        assert np.allclose(route, trace["route_position"], atol=1e-6)
        assert np.isclose(np.abs(route[:, 1]).max(), result["max_abs_lateral_m"], atol=1e-6)
        assert np.allclose(
            np.asarray(result["target_xy"]) - trace["root_position_w"][0, :2],
            rotation @ [protocol.goal_m, 0.0],
            atol=1e-6,
        )
        for index, (xy, height, up, seconds) in enumerate(
            zip(
                trace["root_position_w"][:, :2],
                trace["root_height_m"],
                trace["up_cosine"],
                trace["time_s"],
                strict=True,
            )
        ):
            x, y = xy
            inside = (-1 < x < 1 and -6 < y < 0.5) if condition == "earth" else (-0.5 < x < 6 and -1 < y < 1)
            outcome = classify_sample(xy, result["target_xy"], height, up, seconds, inside, protocol)
            assert outcome == (result["outcome"] if index == count - 1 else None), (path, index, outcome)
        assert result["maximum_soil_mass_error_kg"] < 1e-5
        if condition == "mars":
            assert result["soil_reset_verified"]
        results.append(result)
        trajectories.append(trace)
    assert [r["seed"] for r in results] == manifest["seeds"]
    return manifest, results, trajectories


def compare(root):
    earth_manifest, earth, earth_traces = audit_condition(root, "earth")
    mars_manifest, mars, mars_traces = audit_condition(root, "mars")
    for key in (
        "seeds",
        "protocol",
        "checkpoint_sha256",
        "backend",
        "rigid_substeps",
        "rigid_contacts",
        "physics_dt",
        "control_dt",
        "source_sha256",
        "asset_sha256",
        "air_force_refresh_s",
    ):
        assert earth_manifest[key] == mars_manifest[key], f"Unmatched {key}"
    paired_arrivals, transitions = [], {}
    for e, m, et, mt in zip(earth, mars, earth_traces, mars_traces, strict=True):
        assert e["perturbation"] == m["perturbation"]
        earth_joints = dict(zip(earth_manifest["joint_names"], et["joint_position"][0], strict=True))
        mars_joints = dict(zip(mars_manifest["joint_names"], mt["joint_position"][0], strict=True))
        assert earth_joints.keys() == mars_joints.keys()
        assert all(abs(earth_joints[name] - mars_joints[name]) < 1e-5 for name in earth_joints)
        for r, manifest in ((e, earth_manifest), (m, mars_manifest)):
            rotation = route_rotation(manifest["route_yaw_rad"])
            start = np.asarray(r["start_position_w"][:2]) @ rotation
            velocity = np.asarray(r["initial_velocity_w"][:2]) @ rotation
            assert np.allclose(start, [r["perturbation"]["forward_m"], r["perturbation"]["lateral_m"]], atol=1e-5)
            assert np.allclose(
                velocity,
                [r["perturbation"]["forward_velocity_m_s"], r["perturbation"]["lateral_velocity_m_s"]],
                atol=1e-5,
            )
        key = f"earth_{e['outcome']}__mars_{m['outcome']}"
        transitions[key] = transitions.get(key, 0) + 1
        if e["outcome"] == m["outcome"] == "success":
            paired_arrivals.append(m["duration_s"] - e["duration_s"])
    report = dict(
        complete=True,
        audit="All samples re-scored; paired starts, hashes, seeds, reset and mass checks passed.",
        earth=summarize(earth),
        mars=summarize(mars),
        paired_outcomes=transitions,
        both_success_count=len(paired_arrivals),
        mean_mars_minus_earth_arrival_s=float(np.mean(paired_arrivals)) if paired_arrivals else None,
        protocol=earth_manifest["protocol"],
        limitations=mars_manifest["limitations"],
    )
    report["mars"]["soil_deformation_observed_trials"] = sum(
        bool((np.abs(t["min_soil_displacement_m"]) > 1e-5).any() or (np.abs(t["max_soil_displacement_m"]) > 1e-5).any())
        for t in mars_traces
    )
    report["mars"]["maximum_soil_mass_error_kg"] = max(r["maximum_soil_mass_error_kg"] for r in mars)
    if (root / "repeat-check.json").exists():
        report["repeat_check"] = json.loads((root / "repeat-check.json").read_text())
        report["limitations"].append(
            "Seeded initial states do not guarantee bitwise deterministic GPU MPM trajectories."
        )
    with (root / "comparison.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    plot(root, (earth, mars), (earth_traces, mars_traces), report)
    print(json.dumps(report, indent=2))


def plot(root, results, traces, report):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    colors = {"success": "#0d9488", "fall": "#dc2626", "out_of_bounds": "#d97706", "timeout": "#7c3aed"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, condition, rows, paths in zip(axes, ("earth", "mars"), results, traces, strict=True):
        for row, trajectory in zip(rows, paths, strict=True):
            xy = trajectory["route_position"]
            ax.plot(xy[:, 0], xy[:, 1], color=colors[row["outcome"]], alpha=0.35, linewidth=0.8)
        ax.add_patch(Circle((5, 0), 0.5, fill=False, color="#0f172a", linestyle="--"))
        ax.scatter([0, 5], [0, 0], color="#0f172a", s=20)
        summary = report[condition]
        ax.set(
            title=f"{condition.title()}: {summary['outcomes']['success']}/{summary['trials']} arrivals",
            xlabel="Forward displacement (m)",
            ylabel="Lateral displacement (m)",
            xlim=(-0.2, 6.3),
            ylim=(-1.3, 1.3),
        )
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
    fig.suptitle(
        "Paired H1 trials · 5 m goal · 0.5 m arrival radius\nTeal: arrival · Orange: route exit · Red: fall",
        fontsize=12,
    )
    fig.savefig(root / "trajectories.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for condition, rows, color in zip(("Earth", "Mars"), results, ("#2563eb", "#c2410c"), strict=True):
        arrivals = [r["duration_s"] for r in rows if r["outcome"] == "success"]
        axes[0].hist(
            arrivals, bins=np.arange(5, 31, 0.5), alpha=0.6, label=f"{condition} (n={len(arrivals)})", color=color
        )
        axes[1].hist(
            [r["max_abs_lateral_m"] for r in rows],
            bins=np.linspace(0, 1.3, 27),
            alpha=0.6,
            label=condition,
            color=color,
        )
    axes[0].set(xlabel="Arrival time among successes (s)", ylabel="Trials")
    axes[1].set(xlabel="Maximum absolute lateral drift (m)", ylabel="Trials")
    for ax in axes:
        ax.legend()
        ax.grid(axis="y", alpha=0.15)
    fig.suptitle("Combined environment comparison; assumed soil material, narrow start variations")
    fig.savefig(root / "metrics.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--mars-shards", nargs="+", type=Path)
    args = parser.parse_args()
    if args.mars_shards:
        merge_mars_shards(args.output_dir.resolve(), [p.resolve() for p in args.mars_shards])
    compare(args.output_dir.resolve())
