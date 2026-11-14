"""Run independent physics controls and convergence checks; no graphics or policy.

Exit 0 means these numerical checks passed, NEVER that a Mars material was
validated. Exit 1 records failed checks without hiding them. Each child process
gets a fresh solver; all logs and JSON reports are preserved with unique names.
Run with /home/eddie/env_isaaclab/bin/python from the project root.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def main():
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(ZoneInfo("America/Vancouver")).strftime("%Y%m%d-%H%M%S-%f")
    log_dir, report_dir = root / "outputs/log", root / "outputs/physics"
    log_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    # Each comparison changes exactly one setting from baseline.
    cases = {
        "baseline": [],
        "feedback_off": ["--coupling", "one_way"],
        "timestep_halved": ["--dt", "0.0025"],
        "grid_halved": ["--voxel-size", "0.01"],
        "regularization_reduced": ["--background-regularization", "0.1"],
        "shear_low": ["--shear-force", "5"],
        "shear_high": ["--shear-force", "40"],
    }
    runs = {}
    for name, options in cases.items():
        log = log_dir / f"mpm-suite-{stamp}-{name}.log"
        command = [sys.executable, str(root / "scripts/benchmark_mpm_soil.py"), *options]
        print(f"VERIFY_RUN {name} log={log}", flush=True)
        with log.open("x") as stream:
            result = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=False)
        paths = [
            line.removeprefix("BENCHMARK_SAVED ")
            for line in log.read_text().splitlines()
            if line.startswith("BENCHMARK_SAVED ")
        ]
        runs[name] = {"exit_code": result.returncode, "log": str(log.relative_to(root)), "command": command}
        if paths:
            runs[name]["report"] = str(Path(paths[-1]).relative_to(root))
            runs[name]["data"] = json.loads(Path(paths[-1]).read_text())
        print(f"VERIFY_RESULT {name} exit={result.returncode}", flush=True)

    def good(name):
        return runs[name]["exit_code"] == 0 and runs[name].get("data", {}).get("finite", False)

    checks = {"all_runs_finite": all(good(name) for name in cases)}
    metrics = {}
    if good("baseline"):
        baseline = runs["baseline"]["data"]
        weight = baseline["expected_foot_weight_n"]
        force = baseline["last_quarter_second_mean_soil_force_n"][2]
        metrics["support_relative_error"] = abs(force - weight) / weight
        checks["supports_weight_within_5_percent"] = metrics["support_relative_error"] < 0.05
        free = [s for s in baseline["samples"] if 0 < s["time_s"] < 0.1]
        acceleration = (free[-1]["vz_m_s"] - free[0]["vz_m_s"]) / (free[-1]["time_s"] - free[0]["time_s"])
        metrics["freefall_acceleration_m_s2"] = acceleration
        checks["freefall_matches_gravity"] = abs(acceleration + baseline["parameters"]["gravity"]) < 0.01
        checks["foot_stays_above_rigid_base"] = baseline["final_foot_bottom_m"] > 0.08
        for name in ("timestep_halved", "grid_halved", "regularization_reduced"):
            check = f"{name}_height_difference_under_2mm"
            checks[check] = False
            if good(name):
                delta = abs(runs[name]["data"]["final_foot_bottom_m"] - baseline["final_foot_bottom_m"])
                metrics[f"{name}_height_difference_m"] = delta
                checks[check] = delta < 0.002
        checks["feedback_off_falls_through_soil"] = (
            good("feedback_off") and abs(runs["feedback_off"]["data"]["final_foot_bottom_m"]) < 0.002
        )
        if good("shear_low") and good("shear_high"):
            low, high = runs["shear_low"]["data"], runs["shear_high"]["data"]
            metrics["low_load_displacement_m"] = abs(low["final_foot_x_m"] - baseline["final_foot_x_m"])
            metrics["high_load_displacement_m"] = abs(high["final_foot_x_m"] - baseline["final_foot_x_m"])
            checks["small_shear_resisted"] = (
                metrics["low_load_displacement_m"] < 0.002
                and abs(low["last_quarter_second_mean_soil_force_n"][0] + 5.0) < 0.5
            )
            checks["large_shear_causes_slip"] = metrics["high_load_displacement_m"] > 0.01
    else:
        checks["baseline_valid"] = False

    numerical_pass = all(checks.values())
    output = report_dir / f"mpm-verification-{stamp}.json"
    report = {
        "numerical_checks_passed": numerical_pass,
        "mars_material_validated": False,
        "ready_for_h1_loose_sand_inference": False,
        "threshold_note": "2 mm / 5% are provisional numerical gates, not experimental Mars tolerances.",
        "remaining": [
            "Three-level grid/time convergence",
            "Bed-depth/boundary sensitivity",
            "Calibrated repose, pressure-sinkage and shear curves",
            "Independent material validation",
            "H1 locomotion validation on calibrated soil",
            "Atmosphere and fine dust not modeled",
        ],
        "checks": checks,
        "metrics": metrics,
        "runs": runs,
    }
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("VERIFY_CHECKS " + json.dumps(checks), flush=True)
    print(f"VERIFY_SAVED {output}", flush=True)
    return 0 if numerical_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
