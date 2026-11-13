# Vision for Project: Mars-Project

Run one pretrained H1 walking policy on Earth and Mars, then compare outcomes.

## End goal

1. **Earth:** Walk straight for roughly 5–10 footsteps in the Poly Haven scene.
2. **Mars:** Run the same checkpoint with Mars gravity and, later, dust, reduced drag, and loose regolith.
3. **Findings:** Compare success over 100 trajectories per environment using the same target distance.
4. **Recording:** Save short videos for visual iteration, then produce final comparison footage.

## Current state

The repository cleanup is complete. The next implementation is a fixed-command, timed rollout with video recording.
A successful walking run or recording has **not** been verified.

| Area | Implemented | Still needed |
| --- | --- | --- |
| Robot/task | One `Mars-Project-H1` task inherits Isaac Lab's H1 observations, actions, rewards and PPO configuration. | Verify the cached checkpoint matches the installed task and produces stable walking. |
| Policy | Cached pretrained RSL-RL checkpoint; runner can load it with `--checkpoint`. Without one, it sends zero actions. | Make checkpoint inference the normal walking workflow; no new training is planned. |
| Movement | Preview holds forward, lateral and yaw velocity commands at zero; pushes and observation noise are disabled. | Set a constant forward speed, zero lateral/yaw commands, disable standing-command sampling, and control initial pose/velocity and randomization. |
| Duration | `--steps N` bounds environment control steps; `0` runs until closed. | Add a 5–7 **simulation-second** duration using `env.step_dt`; stop on falling instead of silently continuing through resets. Control steps are not footsteps. |
| Earth | Packaged Verdant Trail USD, collision ground, scene camera and active lighting/exposure setup. | Restore textures: the cleanup audit found all 128 referenced textures were Git LFS pointers. Check sky, ground contact and framing in a rendered run. |
| Mars | Optimized canyon scenery, rigid collision mesh, sky/sun lighting and selectable −3.71 m/s² gravity (Earth: −9.81). | Dust, explicit drag changes and deformable/loose regolith are absent. Current canyon is a rigid visual/terrain approximation. |
| Camera/video | One fixed camera; no keyboard control or camera animation. | Add offscreen rendering over SSH, frame capture and MP4 encoding; save to `/home/eddie/Desktop/` and verify playback, framing and duration. No video was produced by this cleanup. |
| Evaluation | Existing environment termination signals are available. | Log distance, elapsed simulation time and falls; define success and target distance; run 100 seeded trajectories per environment and export results. |

Start with a **6-second rollout**, rather than footstep counting. Counting 5–10 actual footsteps needs additional
contact-event logic. A timed preview also does not replace the shared-distance criterion for the final evaluation.

## Cleanup performed

- Removed disconnected WASD controls and the separate stand-up training task.
- Removed the broken `play_mars.py` shortcut and redundant environment-listing wrapper; `scripts/view_scene.py` is the sole launcher.
- Removed robot POV, orbit controls and the unused Earth dome-orientation helper; retained active sky, exposure and lighting logic.
- Removed the offline canyon builder, original USDZ, intermediate preview mesh and anchor; kept optimized runtime assets and source/build metadata. Rebuilding from raw sources now requires restoring the builder and inputs from version history/source provenance.
- Removed outdated planning/architecture documents; this README describes the current structure and next work.
- Kept every texture referenced by the active USD scenes. Asset validation now checks referenced dependencies for missing files and LFS pointers before simulation startup.
- Kept the pretrained checkpoint cache locally and added it to `.gitignore`.

## Remaining implementation order

1. Restore LFS scene content and verify checkpoint loading and the fixed camera in a short rendered smoke run.
2. Add constant forward motion, reproducible resets, a 6-second limit and stop-on-fall behavior.
3. Add offscreen MP4 capture to the desktop for viewing after an SSH run; inspect the first Earth clip and iterate.
4. Repeat with the same checkpoint on Mars; then add metrics, repeated trials and the extra environmental physics.

## Run the existing preview

Use the existing Isaac Sim environment from the project root:

```bash
/home/eddie/env_isaaclab/bin/python scripts/view_scene.py --appearance earth --steps 300
/home/eddie/env_isaaclab/bin/python scripts/view_scene.py --appearance grand-canyon --gravity mars --steps 300
```

These are stationary previews unless `--checkpoint /absolute/path/to/checkpoint.pt` is supplied;
even with a checkpoint the current velocity command is zero. They use a Kit viewport and are **not yet an
SSH/offscreen recording workflow**. `--steps 300` means 300 control steps, not 300 footsteps or a guaranteed video duration.

Restore missing LFS content from a configured remote with `git lfs pull`, or use `git lfs checkout` when the objects
are already cached. Validation reports missing/unhydrated assets before launching the simulator.

## Project structure

```text
scripts/view_scene.py             Single CLI entry point
src/mars_project/
  scene_runner.py                CLI, simulator lifecycle, checkpoint inference and stepping
  tasks/h1.py                    H1 environment/PPO configuration and Gym registration
  scenes/setup.py                Stationary preview configuration and scene selection
  scenes/earth.py                Earth geometry, lighting and rendering settings
  scenes/canyon.py               Canyon geometry, collision and lighting
  scenes/cameras.py              One fixed camera
  assets/scene_paths.py          Runtime paths and USD dependency validation
  assets/data/                  Packaged runtime assets and provenance
tests/                          Registration and asset-validation checks
```

`h1.py` defines **what environment the policy controls**. `scene_runner.py` defines **how to launch and step it**.
Both are needed; there is no second stand-up task or keyboard-control module.

## Environment and checks

`pyproject.toml` and `uv.lock` describe the environment and editable Isaac Lab dependencies under
`/home/eddie/IsaacLab`. Update `[tool.uv.sources]` if that checkout moves. `uv sync` defaults to Newton;
this project's scene runner explicitly uses Isaac Sim PhysX and Kit, available in the existing interpreter above.
The optional dependency extras remain available for Isaac Lab compatibility.

```bash
.venv/bin/python -m pytest -q
```

The tests check registration and asset validation; they do not establish walking stability or rendering correctness.
