# Mars-Project

Run the same pretrained Unitree H1 walking policy in Earth and Mars simulation environments, record its motion, and compare measured outcomes.

The project combines Isaac Lab, Newton/MuJoCo Warp, deformable soil, atmospheric forces, and a repeatable evaluation pipeline. The **100-trial evaluation for each environment is complete**. The simulation uses fixed policy weights; no policy training was performed in this project. Its soil and dust models remain experimental.

- **[Findings report](report.md):** experiment setup, results, interpretation, and validation.
- **[Physics sources and assumptions](sources.md):** material parameters, atmosphere, approximations, and references.
- **[Recorded results](outputs/evaluations/paired-100-20261114-1048/):** trajectories, metrics, plots, and input snapshots from 14 November 2026.

Jump to [repo layout](#repository-organization), [setup](#setup), [previews and recording](#run-a-preview), [evaluation](#reproduce-the-trajectory-evaluation), [outputs](#outputs-and-reports), or [lighting troubleshooting](#camera-and-lighting-troubleshooting).

## Results

| Metric | Earth | Mars |
| --- | ---: | ---: |
| Successful arrivals | **89/100** | **73/100** |
| Boundary exits | 11 | 27 |
| Falls / timeouts / invalid states | 0 / 0 / 0 | 0 / 0 / 0 |
| Mean arrival time among successful trials | 10.08 s | 12.44 s |
| Mean per-trial peak absolute lateral drift, all trials | 0.320 m | 0.434 m |

Both conditions succeeded for 70 paired seeds; Mars took **2.42 s longer on average within those pairs**. All 200 trajectories, containing 119,327 saved samples, passed the full audit. Every Mars trial showed soil deformation and passed reset and mass-conservation checks.

See the [machine-readable comparison](outputs/evaluations/paired-100-20261114-1048/comparison.json), [trajectory plot](outputs/evaluations/paired-100-20261114-1048/trajectories.png), and [metric distributions](outputs/evaluations/paired-100-20261114-1048/metrics.png). The [findings report](report.md) includes uncertainty intervals and the execution-status caveat: the final worker returned exit code 143 after saving its complete batch; every saved trial passed audit.

These results compare **combined environments** with different source terrains, gravity, soil, and atmosphere. They do not isolate the effect of gravity or establish real-world Mars walking reliability.

## What is simulated

| Component | Earth | Mars |
| --- | --- | --- |
| Scenery | Poly Haven Verdant Trail | Packaged Grand Canyon terrain |
| Gravity | 9.81 m/s² downward | 3.71 m/s² downward |
| Walking surface | Rigid, uneven local collision mesh | 4 cm of deformable sand over a rigid substrate |
| Robot air drag | Earth reference atmosphere | Mars reference atmosphere |
| Default visual-preview backend | PhysX | Newton/MuJoCo Warp coupled to implicit MPM |
| Measured evaluation backend | Newton/MuJoCo Warp | Newton/MuJoCo Warp coupled to implicit MPM |

The robot uses a cached **rough-terrain H1 checkpoint** with 256 observations and 19 joint-position actions. Joint ordering and actuator gains are preserved when switching environments. The registered task is `Mars-Project-H1`, built on Isaac Lab's rough-H1 task configuration.

The main environment features are:

- **Deformable soil:** Newton's material point method (MPM) exchanges forces with the robot. The local 8.5 × 4 m soil domain contains 170,000 material samples. Nearby and disturbed soil is simulated actively; settled deformation persists.
- **Shared terrain representation:** the visible soil surface and policy height observations use the same volume-based reconstruction. Detailed scenery is separate from the bounded collision corridor used for walking.
- **Airborne dust:** a small fraction of detached material becomes mass-carrying parcels subject to gravity and gas drag, then deposits back onto the surface. Kit renders the suspended material as a volume.
- **Robot air resistance:** quadratic drag acts on individual links using their velocity and projected area. Atmospheric forces are separate from the robot's motor gains.

Parameter values and model limits are documented in [sources.md](sources.md).

## Repository organization

```text
src/mars_project/           Reusable simulation code
  tasks/                   H1 control configuration and task registration
  scenes/                  World assembly, scenery, cameras, and lighting
  environment/             Active soil, dust, and aerodynamic models
  assets/                  Packaged geometry/textures, validation, and terrain sampling
  scene_runner.py          Preview lifecycle and checkpoint playback
  evaluation.py            Trial protocol, paired starts, and outcome scoring
  screenshots.py           Screenshot capture and output routing
  recordings.py            MP4 encoding and smooth orbit tracking
scripts/                   Launchers, experiments, audits, and asset builders
tests/                     Automated checks for the active implementation
outputs/                   Generated evidence, ignored by Git
report.md                  Completed evaluation findings
sources.md                 Physics assumptions and supporting references
AGENTS.md                  Repository editing and output conventions
pyproject.toml / uv.lock    Dependencies and local Isaac Lab integration
```

The separation keeps **robot control**, **world physics**, and **experiment orchestration** independently understandable. Both preview and evaluation reuse the simulation code. Evaluation runs without rendering and saves raw trajectories; a separate audit reads those artifacts to produce the comparison. Videos provide visual evidence, while the evaluation pipeline supplies measured outcomes.

## Setup

The existing workstation uses Linux, Python 3.12, an NVIDIA GPU, and a local Isaac Lab checkout at `/home/eddie/IsaacLab`. The completed evaluation used an RTX 5080, PyTorch 2.11 with CUDA 13, and Newton 1.6.1; the complete [software snapshot](outputs/evaluations/paired-100-20261114-1048/software.json) is saved with the results.

Two Python environments serve different purposes:

| Environment | Use |
| --- | --- |
| `/home/eddie/env_isaaclab/bin/python` | Existing simulator runtime, rendered previews, and GPU experiments |
| `.venv/bin/python` | Repository development and unit tests |

For a new checkout, install the matching Isaac Lab/Isaac Sim runtime and adjust the editable dependency paths in `pyproject.toml` if the Isaac Lab checkout moves. `uv sync --group dev` prepares the development environment from the checked-in configuration; it does not replace simulator installation.

From the project root, prepare the shell used by the commands below:

```bash
git lfs pull
mars_python=/home/eddie/env_isaaclab/bin/python
mkdir -p outputs/log
```

Supply the matching rough-H1 checkpoint at `.pretrained_checkpoints/h1_rough.pt`. The checkpoint, Python environments, and generated outputs are ignored by Git. A fresh clone therefore does not contain the saved runs or checkpoint. The flat-H1 checkpoint has an incompatible observation layout.

Asset validation rejects missing files, unresolved USD references, and unhydrated Git LFS pointers before preview startup. MP4 encoding also requires `imageio-ffmpeg` in the simulator interpreter; if missing, install it with `uv pip install --python "$mars_python" imageio-ffmpeg`.

## Run a preview

These commands open a Kit viewport and run six simulated seconds of forward walking. Run them in a desktop session with a display:

```bash
"$mars_python" scripts/view_scene.py \
  --appearance earth --steps 300 \
  --checkpoint .pretrained_checkpoints/h1_rough.pt --forward-velocity 0.5 \
  > "outputs/log/earth-preview-$(date +%Y%m%d-%H%M%S-%N).log" 2>&1

"$mars_python" scripts/view_scene.py \
  --appearance grand-canyon --steps 300 \
  --checkpoint .pretrained_checkpoints/h1_rough.pt --forward-velocity 0.5 \
  > "outputs/log/mars-preview-$(date +%Y%m%d-%H%M%S-%N).log" 2>&1
```

| Option | Behavior |
| --- | --- |
| `--appearance earth` / `grand-canyon` | Selects the world; default gravity follows the selected appearance. |
| `--gravity earth` / `mars` | Overrides gravity independently of scenery. |
| `--soil rigid` | Selects the rigid-ground PhysX baseline for canyon previews. |
| `--air-drag auto` / `earth` / `mars` / `off` | Selects robot air drag. `auto` follows gravity; dust atmosphere continues to follow gravity. |
| `--forward-velocity 0.5` | Commands body-forward motion in m/s; requires a checkpoint. Default is zero. |
| `--steps N` | Limits control steps; at the current 20 ms control interval, 300 steps is six simulated seconds. Zero runs until the window closes. |

Without a checkpoint, the launcher sends zero actions. A loaded checkpoint with zero velocity command is a stationary policy preview. Preview autoresets remain enabled and are logged; a completed video is not itself a scored successful trajectory.

### Record videos and screenshots over SSH

For an explicitly requested six-second orbit recording:

```bash
"$mars_python" scripts/view_scene.py \
  --appearance earth --steps 300 \
  --checkpoint .pretrained_checkpoints/h1_rough.pt --forward-velocity 0.5 \
  --record earth-demo --camera-motion orbit --capture-requested \
  --kit_args=--no-window \
  > "outputs/log/earth-video-$(date +%Y%m%d-%H%M%S-%N).log" 2>&1
```

Use `--appearance grand-canyon --record mars-demo` for Mars. Videos are 1080p, 25-fps H.264 MP4s with QuickTime-compatible encoding. Orbit mode adds one unrecorded simulation second for settling; render warmup does not advance physics. `--kit_args=--no-window` still uses the GPU and Kit renderer; SSH does not stream the viewport.

For stills, use `--screenshot LABEL` with a positive step limit. `--screenshot-pair LABEL` captures two frames one simulation second apart; `--screenshot-series LABEL --screenshot-count 5` captures a series. Capture modes are mutually exclusive. `--screenshot-requested` and `--capture-requested` are aliases.

Only mark captures as requested when the user explicitly asked for them. Otherwise omit that flag so diagnostics use `recordings/misc/`. Requested labels such as `earth-demo` and `mars-demo` receive their own environment-grouped folders. Captures use Vancouver dates and avoid overwriting existing files.

Example visual previews: [Earth orbit](outputs/recordings/11-14-2026/earth/earth-air-resistance/10:07am_earth-air-resistance-orbit.mp4) and [Mars orbit](outputs/recordings/11-14-2026/mars/mars-air-resistance/10:09am_mars-air-resistance-orbit.mp4). These recordings are separate from the headless evaluation batch.

## Reproduce the trajectory evaluation

The evaluator uses the same checkpoint and rigid-body solver for both conditions. Each trial starts from a reset state, records every control step, and stops on its first terminal outcome. Autoreset is disabled so it cannot replace the terminal state.

| Protocol | Setting |
| --- | --- |
| Trials | 100 per condition, paired seeds 42–141 |
| Target | 5 m along the nominal route from the actual starting position |
| Arrival | Within 0.5 m horizontally while upright, by 30 s |
| Start variations | Position ±5 cm, heading ±3°, horizontal velocity ±0.05 m/s, yaw rate ±0.02 rad/s |
| Control | Constant 0.5 m/s body-forward command; no heading feedback or observation noise |
| Timing | 5 ms physics ticks, two rigid substeps, 20 ms control steps; timing includes landing |
| Fall | Base below 0.55 m above local terrain or tilt exceeding 60° |
| Boundary | Leaving the guarded collision corridor is a separate failure |

Run both conditions sequentially in a **new output directory**. The subshell stops on a failed command:

```bash
(
  set -e
  run_id="$(date +%Y%m%d-%H%M%S-%N)"
  result_dir="outputs/evaluations/paired-100-$run_id"
  for condition in earth mars; do
    "$mars_python" -u scripts/evaluate_trajectories.py \
      --condition "$condition" --trials 100 --seed 42 \
      --output-dir "$result_dir" \
      > "outputs/log/evaluation-$condition-$run_id.log" 2>&1
  done
  "$mars_python" scripts/summarize_evaluation.py "$result_dir" \
    > "outputs/log/evaluation-audit-$run_id.log" 2>&1
)
```

For a small smoke evaluation, use `--trials 2` for both conditions. The full audit checks all saved samples, terminal outcomes, paired starts, seeds, and matching input hashes. It rejects incomplete batches and writes `comparison.json`, `trajectories.png`, and `metrics.png`. Existing reports are preserved; use fresh directories for new runs.

The original completed batch remains in `outputs/evaluations/paired-100-20261114-1048/`. Its manifests and input snapshots preserve the exact evaluation-time sources and checkpoint. Later source cleanup does not rewrite that evidence. GPU MPM motion is not guaranteed bitwise deterministic even with identical seeded starts.

## Outputs and reports

| Location | Contents |
| --- | --- |
| `outputs/evaluations/<run>/` | Per-condition trial JSONs and NPZ trajectories, manifests, comparison metrics, and scientific plots |
| `outputs/physics/` | Isolated solver benchmarks and numerical verification reports |
| `outputs/log/` | All run, test, simulator, and diagnostic `.log` files |
| `outputs/recordings/MM-DD-YYYY/` | Explicitly requested captures; named Earth/Mars experiments are grouped under `earth/` or `mars/` |
| `outputs/recordings/misc/MM-DD-YYYY/` | All unrequested diagnostic screenshots and videos |
| `outputs/lighting/` | Standalone lighting comparisons, sky images, and render-reference PNG assets |

Each evaluation trial has a JSON outcome record and a compressed NPZ with base pose/velocity, joint positions, actions, foot positions, route coordinates, drag force, and soil-surface extrema. The completed dataset also includes software versions, source/checkpoint hashes, input snapshots, and execution status. Preliminary checks and original worker batches are preserved separately.

Use the capture pipeline for simulation images and videos. Its recording paths take precedence over `outputs/lighting/`. Keep loose logs and PNGs out of the `outputs/` root, preserve requested folder names, and use unique names for new artifacts. See [AGENTS.md](AGENTS.md) for the exact conventions. Artifact links in this README require the local outputs, which are not distributed through Git.

## Tests and verification tools

Run the fast checks from the project root:

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src scripts tests
```

After removal of the retired implementations, **33 tests and Ruff passed**. A [fresh paired smoke evaluation](outputs/evaluations/cleanup-smoke-20261114-174943-120051636/comparison.json) also passed audit for two Earth and two Mars trials. The original full evaluation had 42 passing tests before the obsolete tests were removed; that historical count remains in its report.

| Script | Purpose |
| --- | --- |
| `check_mpm_playback.py` | Loaded soil/robot interaction, live height sensing, and reset; optionally checks rendered surface updates |
| `check_soil_walk.py` | Seven-second policy run with finite-state, reset, and soil/dust mass checks |
| `check_air_drag.py` | Applied force versus measured momentum, force composition, and reset on Newton or PhysX |
| `benchmark_mpm_soil.py` | Isolated loaded-foot experiment, independent of the walking policy |
| `verify_mpm_soil.py` | Support, shear, coupling, timestep, and grid-refinement comparisons |
| `build_earth_collision.py`, `build_mars_collision.py` | Rebuild local collision surfaces from packaged scene geometry |
| `build_sand_normal.py` | Rebuild the normal texture used by the active soil surface |

These scripts live under `scripts/`. Use the simulator interpreter for GPU checks and redirect console logs to unique files in `outputs/log/`. For example:

```bash
"$mars_python" scripts/check_soil_walk.py --visualizer none \
  > "outputs/log/soil-walk-check-$(date +%Y%m%d-%H%M%S-%N).log" 2>&1
```

The independent MPM verification suite still has unresolved numerical convergence checks. A failed convergence gate is recorded evidence, not a failed trajectory audit; even a passing numerical suite would not establish experimentally calibrated Mars soil.

## Camera and lighting troubleshooting

**Earth illumination can fail intermittently at startup**, producing a near-black robot or ground despite unchanged assets and settings. Successful retries support a timing-related explanation, but the precise RTX mechanism remains unconfirmed. The issue is not considered permanently fixed.

The runner follows this startup sequence; preserve it when changing cameras, exposure, lighting, or renderer initialization:

1. Configure synchronous USD/material loading and Earth renderer settings **before `gym.make()`**. For MPM playback, select the soil renderer before scene creation.
2. After Earth environment creation, apply the initial view settings and render three updates **before reset or camera switching**. Retain the initial view setup as well as final-camera setup.
3. Finish scene setup and reset the environment, then select the final renderer and create the final camera. Author its exposure and render-product settings **before `set_active_camera()`**. HDR dome sampling must be authored on the render product as `omni:rtx:lights:dome:samplingStrategy = imageBasedLighting`, using `OmniRtxSettingsCommonAdvancedAPI_1`, as well as the global Carb setting.
4. Activate the camera, render three updates without advancing physics, and log the effective `RENDER_VIEW` state. Captures then receive 60 render-only warmup updates.

Preserve source HDR intensity, neutral exposure, both Earth dome assets, and ACES. The configured RTX ISO value of zero bypasses its camera multiplier; it is not a physical camera ISO. Fixed captures and orbits frame the robot at its body midpoint; changing camera height must not change illumination settings.

Rigid previews use real-time path tracing; MPM previews use path tracing for heterogeneous dust volumes. Keep the Kit `omni.volume` writer matched to its renderer: the previously tested Warp NanoVDB writer produced an incompatible volume version.

After a rendering change, inspect robot and ground illumination in **at least two fresh-process captures**, with separate logs. Settings dumps and unit tests cannot prove the image is correct. Preserve failed captures, compare fixed-pose renders, and investigate loading order before changing exposure. Diagnostic captures belong in `outputs/recordings/misc/MM-DD-YYYY/`.

## Scope and limitations

- The completed comparison measures this policy on these two combined environments. Terrain, gravity, soil, and atmosphere vary together.
- The shallow soil layer, material parameters, local active-region boundaries, and dust release model are assumptions. Material calibration and grid/timestep convergence remain unresolved.
- Surface reconstruction assumes approximately constant bulk density and does not represent overhangs or full elastic volume changes. MPM still computes three-dimensional contact forces.
- Robot drag uses projected bounding-box areas, an assumed drag coefficient of one, and prescribed still air. The model does not solve atmospheric flow or wake shielding.
- Small starting-state variations cover a narrow operating range. Falls use height/tilt thresholds; contact-based slip is not measured.
- Source scenery extends beyond the physical walking corridor. Keep robot motion and height sensing within the collision domain.

See [report.md](report.md) for the measured findings and [sources.md](sources.md) for the physical assumptions. Code licensing is recorded in [LICENSE](LICENSE); scene provenance is retained with the [Earth assets](src/mars_project/assets/data/polyhaven/verdant_trail/download-source.json) and [canyon assets](src/mars_project/assets/data/marslab/grand_canyon/download-source.json).
