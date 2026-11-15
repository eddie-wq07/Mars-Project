# Earth and Mars Trajectory Test — Findings Report

**Evaluation date:** 14 November 2026

**Status:** Complete — 100 Earth trajectories and 100 Mars trajectories audited.

## 1. Overview

The same pretrained rough-H1 walking policy reached the target in **89% of Earth trials** and **73% of Mars trials**. Mean arrival time among successful trials was **10.08 s on Earth** and **12.44 s on Mars**. Every unsuccessful trial exited the route boundary; no falls, timeouts, or invalid states were recorded.

These findings describe the tested simulation environments and small starting-state variations. Soil calibration remains unresolved, so they do not establish real-world Mars walking performance.

## 2. Test Setup

- **Conditions:** rigid Earth trail and Mars canyon with a shallow, 4 cm MPM soil layer. Both used the Newton/MuJoCo Warp rigid-body solver.
- **Policy:** identical pretrained rough-H1 checkpoint, fixed weights, and a constant 0.5 m/s body-forward command. No training or heading feedback was used.
- **Target:** 5 m along the nominal route from each trial's starting position. Arrival required entering a 0.5 m horizontal radius while upright, within 30 simulated seconds.
- **Paired starts:** seeds 42–141 selected matching route-relative variations: position ±5 cm, heading ±3°, horizontal velocity ±0.05 m/s, and yaw rate ±0.02 rad/s. Joint starts were fixed.
- **Timing:** 5 ms physics steps, two rigid substeps, and 20 ms control steps. Trajectories were recorded at 50 Hz; arrival time includes initial landing.
- **Failure criteria:** a fall meant base height below 0.55 m above local terrain or tilt exceeding 60°. Leaving the guarded route counted separately as a boundary exit. Mars soil and dust states were reset before every trial.

## 3. Results

| Metric | Earth | Mars |
| --- | ---: | ---: |
| Trials completed | 100 | 100 |
| Successful arrivals | 89 (89%) | 73 (73%) |
| Boundary exits | 11 | 27 |
| Falls | 0 | 0 |
| Timeouts / invalid states | 0 / 0 | 0 / 0 |
| Successful arrival time, mean ± sample SD | 10.084 ± 0.246 s | 12.440 ± 0.273 s |
| Successful arrival time, median | 10.06 s | 12.38 s |
| Mean per-trial maximum absolute lateral drift | 0.320 m | 0.434 m |
| Arrival-rate Wilson 95% interval | 81.4–93.7% | 63.6–80.7% |

Arrival-time statistics include successful trials only. Lateral-drift statistics include all 100 trials in each condition.

## 4. Findings

1. **Earth had a 16 percentage-point higher arrival rate.** The difference was in route completion: all failures were boundary exits, with no falls under the stated criteria.
2. **Mars was slower in matched successful trials.** Both conditions reached the goal for 70 paired seeds. Within those pairs, Mars took **2.421 s longer on average**. Nineteen seeds succeeded only on Earth, three only on Mars, and eight exited the boundary in both conditions.
3. **Mars showed more lateral drift.** Its mean per-trial peak absolute drift was approximately **0.114 m greater**. This is an observed difference between the combined environments; the test does not identify a single physical cause.
4. **Soil deformation occurred in all 100 Mars trials.** Every Mars reset check passed, and the maximum recorded soil mass-ledger error was **0 kg**. These checks support implementation consistency, but do not validate the assumed soil material.

## 5. Validation and Execution Status

All **200 trajectories and 119,327 saved samples** passed the full audit. Checks covered sample-by-sample outcomes, paired initial states, seeds, matching source/checkpoint/terrain hashes, and reset and mass checks. The evaluation's **42 unit tests and Ruff checks passed**, and both comparison plots were visually inspected.

The final Mars worker reported **exit code 143** after writing all 33 assigned trial pairs, its summary, and the `EVALUATION_COMPLETE` marker. The termination cause was not established. All saved trials passed the full audit; the status is preserved in [execution-status.json](outputs/evaluations/paired-100-20261114-1048/execution-status.json).

## 6. Limitations

- **Combined environment comparison:** source terrain, gravity, soil, and atmosphere differ between conditions. The results do not isolate gravity or air resistance.
- **Experimental soil:** material parameters remain assumed, and grid/timestep convergence and physical calibration remain unresolved.
- **Narrow test coverage:** small initial-state variations do not establish general reliability across terrains, commands, or disturbances.
- **Measurement scope:** falls use height and tilt thresholds. Foot positions were saved, but contact-based slip was not measured.
- **GPU repeatability:** seeded starts do not guarantee bitwise deterministic MPM motion. A repeated Mars seed arrived at 12.18 s and 12.14 s; both runs succeeded.

## 7. Saved Results

All paths below are relative to the project root. The dataset is stored in `outputs/evaluations/paired-100-20261114-1048/`.

- [Full comparison JSON](outputs/evaluations/paired-100-20261114-1048/comparison.json)
- [Trajectory plot](outputs/evaluations/paired-100-20261114-1048/trajectories.png) and [metric distributions](outputs/evaluations/paired-100-20261114-1048/metrics.png)
- [Earth trial data](outputs/evaluations/paired-100-20261114-1048/earth/) and [Mars trial data](outputs/evaluations/paired-100-20261114-1048/mars/): 100 result JSONs and 100 trajectory NPZs per condition
- [Final audit log](outputs/log/evaluation-final-audit-20261114-1334.log)

Software versions, hashed input snapshots, the checkpoint, and the analysis source are preserved with the dataset. The original Mars worker batches are retained separately. The evaluation changes remain uncommitted.
