# Vision for Project: Mars-Project

Run one pretrained H1 walking policy on Earth and Mars, then compare outcomes.

## End goal

1. **Earth:** Walk straight for roughly 5–10 footsteps in the Poly Haven scene.
2. **Mars:** Run the same checkpoint with Mars gravity and, later, dust, reduced drag, and loose regolith.
3. **Findings:** Compare success over 100 trajectories per environment using the same target distance.
4. **Recording:** Save short videos for visual iteration, then produce final comparison footage.

## Flowchart

`src/` holds the importable `mars_project` package. **`tasks/` defines the robot's learning/control environment**
(observations, actions, rewards and resets, mostly inherited from Isaac Lab). **`scenes/` configures its surroundings
and camera** and applies rendering settings. **`scene_runner.py` coordinates both and owns the simulation loop.**

Read downward in execution order. Blue boxes carry the main project logic; gray boxes are supporting steps.
Paths below are relative to `src/mars_project/`, except the launcher in `scripts/`.

```mermaid
flowchart TD
    entry["scripts/view_scene.py<br/>Run the command"]
    path["sys.path.insert(0, .../src)<br/>Tell Python to search src first for imports"]
    main["scene_runner.py → main()<br/>Read CLI options with parse_args()"]
    assets["assets/scene_paths.py → validate_assets()<br/>Check the selected scene's files"]
    task["tasks/__init__.py imports tasks/h1.py<br/>Register Mars-Project-H1 and its H1/PPO configs"]
    config["scene_runner.py → parse_env_cfg()<br/>Load the registered config; select gravity"]
    setup["scenes/setup.py → configure_preview()<br/>Zero velocity commands; disable pushes and observation noise"]
    scene["scenes/earth.py OR scenes/canyon.py<br/>configure_earth_scene() / configure_canyon_scene()<br/>Configure scenery, collision ground, lighting and view"]
    launch["scene_runner.py<br/>launch_simulation() starts the simulator<br/>gym.make() builds the configured environment"]
    finish["scenes/earth.py OR scenes/canyon.py<br/>Apply render settings and verify scene setup"]
    camera["scene_runner.py → env.reset()<br/>scenes/cameras.py → create_camera()<br/>Select the fixed viewport camera"]
    policy["scene_runner.py<br/>If --checkpoint: load RSL-RL inference policy<br/>Otherwise: use zero actions"]
    loop{"scene_runner.py<br/>Simulator running and below --steps limit?"}
    step["scene_runner.py → action_step()<br/>Policy turns observations into actions, or use zeros<br/>env.step(actions) advances the environment<br/>Keep new observations; reset policy state for ended episodes"]
    close["scene_runner.py → env.close()<br/>Exit simulator context"]

    entry --> path --> main --> assets --> task --> config --> setup --> scene
    scene --> launch --> finish --> camera --> policy --> loop
    loop -->|Yes| step
    step -->|Increment step counter| loop
    loop -->|No| close

    classDef core fill:#dbeafe,stroke:#2563eb,color:#172554;
    classDef support fill:#f3f4f6,stroke:#9ca3af,color:#111827;
    class main,task,config,setup,scene,launch,finish,policy,loop,step core;
    class entry,path,assets,camera,close support;
```

`sys.path.insert` does not start the simulator or copy files: it makes `src/mars_project` discoverable by Python.
`from mars_project.scene_runner import main` loads that module and makes its `main` function available;
the launcher's `main()` call then starts the sequence above. Importing `tasks` also registers the task name with Gym.

The arrows show execution order, not that each file calls the next directly: **`main()` calls the helpers and resumes
when each returns**. `configure_preview()` itself calls the selected scene's configuration function. The nested
`action_step()` function runs only when the loop calls it; Isaac Lab's `env.step()` handles the underlying physics,
observations, rewards and episode resets. The project runner does not implement those mechanics itself.

## Current state

Cleanup is complete. **Walking stability and video recording are not yet verified.**
Green = code/assets present, amber = blocked or needs verification, blue = implementation still needed.

```mermaid
flowchart TD
    base["1. PRESENT: H1 task + simulator runner<br/>Inherited observations, actions, rewards and resets<br/>Preview disables pushes and observation noise"]
    earth["2. EARTH: scene + collision ground + lighting present<br/>RESTORED: all 128 texture LFS pointers hydrated and verified<br/>Verify rendered sky and ground contact"]
    mars["2. MARS: canyon + rigid collision mesh + lighting present<br/>Gravity selectable: Earth −9.81 / Mars −3.71 m/s²"]
    camera["3. PRESENT: scene finishing + one fixed camera<br/>Verify framing in a rendered run"]
    policy["4. PRESENT: cached checkpoint + optional policy loading<br/>VERIFY: checkpoint compatibility and stable walking<br/>Without checkpoint: zero actions"]
    loop["5. PRESENT: action_step() + env.step() loop<br/>Velocity commands currently zero<br/>--steps limits control steps; 0 runs until closed"]
    walk["6. ADD: constant forward command + reproducible start<br/>Zero lateral/yaw commands; disable standing sampling<br/>Control reset pose, velocity and randomization<br/>Use pretrained inference by default; no new training"]
    timing["7. ADD: 6 simulation-second limit using env.step_dt<br/>Stop on fall; avoid continuing through episode resets"]
    video["8. ADD: offscreen rendering + frame capture + MP4<br/>Save to /home/eddie/Desktop/ for viewing after SSH runs<br/>Verify playback, framing and duration"]
    iterate["9. NEXT MILESTONE: inspect first Earth walking clip<br/>Then repeat on Mars with the same checkpoint"]
    physics["10. LATER: Mars environmental physics<br/>Dust, reduced drag and loose regolith are absent"]
    evaluation["11. LATER: evaluation and exported results<br/>Define shared distance and success criteria<br/>Log distance, simulation time and falls<br/>Run 100 seeded trajectories per environment"]

    base --> earth
    base --> mars
    earth --> camera
    mars --> camera
    camera --> policy --> loop --> walk --> timing --> video --> iterate
    iterate --> physics --> evaluation

    classDef present fill:#dcfce7,stroke:#16a34a,color:#14532d;
    classDef verify fill:#fef3c7,stroke:#d97706,color:#78350f;
    classDef todo fill:#dbeafe,stroke:#2563eb,color:#172554;
    class base,mars,loop present;
    class earth,camera,policy verify;
    class walk,timing,video,iterate,physics,evaluation todo;
```

**Control steps ≠ footsteps.** Counting 5–10 footsteps requires contact-event logic; start with a 6-second clip.
The final comparison still needs a shared walking distance. No video has been produced yet.

## Cleanup performed

- Removed disconnected WASD controls and the separate stand-up training task.
- Removed the broken `play_mars.py` shortcut and redundant environment-listing wrapper; `scripts/view_scene.py` is the sole launcher.
- Removed robot POV, orbit controls and the unused Earth dome-orientation helper; retained active sky, exposure and lighting logic.
- Removed the offline canyon builder, original USDZ, intermediate preview mesh and anchor; kept optimized runtime assets and source/build metadata. Rebuilding from raw sources now requires restoring the builder and inputs from version history/source provenance.
- Removed outdated planning/architecture documents; this README describes the current structure and next work.
- Kept every texture referenced by the active USD scenes. Asset validation now checks referenced dependencies for missing files and LFS pointers before simulation startup.
- Kept the pretrained checkpoint cache locally and added it to `.gitignore`.

## Remaining implementation order

1. LFS scene content is restored and asset validation passes. Verify checkpoint loading and the fixed camera in a short rendered smoke run.
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

On 2026-11-12, restored all 128 Earth texture placeholders from the local LFS cache. All 136 tracked LFS
assets match their recorded sizes and SHA-256 hashes; Earth and canyon dependency validation and all three
tests pass. Rendered appearance, ground contact and walking stability still require a simulator smoke run.

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

## Render screenshots

Save a PNG at the viewport render resolution (opens in macOS Preview) with local date, time, microseconds and timezone in its filename:

```bash
/home/eddie/env_isaaclab/bin/python scripts/view_scene.py --appearance grand-canyon --steps 1 --screenshot --kit_args=--no-window
```

Screenshots are opt-in: normal runs capture nothing. The command above runs one control step, followed by
60 render-only warmup frames and capture. `--screenshot` requires a positive `--steps` limit.
Optionally use `--screenshot /path/to/image.png` to choose the destination.
Files go to `outputs/screenshots/render_YYYY-MM-DD_HH-MM-SS-microseconds±timezone.png`.
Use `--appearance earth`, `--steps N`, or `--checkpoint /path/to/policy.pt` to override preview settings.
The no-window option still requires an NVIDIA GPU and the Kit renderer; it captures the rendered viewport, not the desktop UI.
The canyon uses our generated orange sky/sun texture; the separate directional light supplies shadows.
