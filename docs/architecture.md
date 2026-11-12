# Mars-Project map

## How the pieces connect

```mermaid
flowchart TD
    config[pyproject.toml<br/>dependencies, Isaac Lab paths, task entry point]
    lock[uv.lock<br/>resolved versions]
    uv[uv sync]
    env[Project Python environment]
    pkg[src/mars_project<br/>installable package]
    tasks[tasks/<br/>Gymnasium task registrations]
    assets[assets/data/<br/>robot and world files]
    isaac[Isaac Lab<br/>loads registered tasks]
    list[scripts/list_envs.py<br/>lists environments]
    tests[tests/<br/>pytest checks]
    hygiene[.gitignore · .gitattributes<br/>pre-commit rules]
    editor[.vscode/<br/>optional editor setup]

    config --> uv
    lock --> uv
    uv --> env
    config -->|editable paths + entry point| pkg
    env --> pkg
    tasks --> pkg
    assets --> pkg
    pkg -->|mars_project.tasks| isaac
    list --> isaac
    tests -->|check| pkg
    hygiene -. applies to .-> pkg
    editor -. supports editing .-> pkg
```

## Project files

```text
Mars-Project/
├── src/mars_project/                 # Importable Isaac Lab extension.
│   ├── __init__.py                   # Package marker.
│   ├── assets/
│   │   ├── __init__.py               # Defines the stable asset-data path.
│   │   ├── scene_paths.py            # Asset paths and validation.
│   │   └── data/                     # Earth and Grand Canyon assets.
│   ├── scene_runner.py               # Owns the simulation loop.
│   ├── scenes/                       # Scenery, camera and preview setup.
│   └── tasks/
│       ├── __init__.py               # Loads project task modules.
│       └── h1.py                     # H1 task, registration and PPO config.
├── tests/
│   └── test_registration.py          # Checks task-registration entry points.
├── scripts/
│   ├── list_envs.py                  # Lists registered environments.
│   └── view_scene.py                 # Launches the scene viewer.
├── docs/
│   ├── architecture.md               # This project map.
│   └── plan.md                       # Mars locomotion and training plan.
├── .vscode/                          # Optional editor setup and templates.
├── pyproject.toml                    # Package metadata, dependencies, paths, and tool settings.
├── uv.lock                           # Resolved dependency versions for uv.
├── README.md                         # Setup and usage instructions.
├── LICENSE                           # Project license terms.
├── .gitignore                        # Excludes local environments, caches, and outputs from Git.
├── .gitattributes                    # Git LFS and file handling rules.
└── .pre-commit-config.yaml           # Checks that run before commits.
```

## What VS Code hides

I found VS Code `settings.json` files rather than a `workspace.json`:

- [`../.vscode/settings.json`](../../.vscode/settings.json) is the Simulation workspace setting. It hides `.git`, `.venv`, caches, `.vscode`, `.gitattributes`, `.gitignore`, `.pre-commit-config.yaml`, `LICENSE`, `README.md`, `architecture.md`, and `plan.md`. It explicitly keeps `pyproject.toml` and `uv.lock` visible.
- [`.vscode/settings.json`](../.vscode/settings.json) applies when Mars-Project itself is opened as the workspace. It hides the same general setup and cache files, and also hides `pyproject.toml` and `uv.lock`.

These `files.exclude` rules only hide items in the VS Code Explorer. They do not delete files, change Python imports, or determine what Git commits.

```mermaid
flowchart LR
    workspace[Simulation settings.json] -->|files.exclude| explorer[VS Code Explorer]
    project[Mars-Project settings.json] -->|files.exclude when opened directly| explorer
    gitignore[.gitignore] -->|Git ignores| git[Git status and commits]
    explorer --> hidden[Hidden from view only]
    hidden -. files remain on disk .-> disk[Project files]
```

**Current state:** `Mars-Project-H1` is registered locally. Earth and Grand Canyon assets live inside this project. Isaac Lab and Isaac Sim remain dependencies, but the sibling Mars Locomotion checkout is no longer used.

## Scene viewer

```mermaid
flowchart TD
    launcher["scripts/view_scene.py · command entry"] --> runner["scene_runner.py · simulation loop"]
    runner --> task["tasks/h1.py · robot task and PPO config"]
    runner --> scenes["scenes/ · Earth, canyon, cameras and scene setup"]
    scenes --> paths["assets/scene_paths.py · asset locations"]
    paths --> data["assets/data/ · Earth and Grand Canyon files"]
    task --> lab["Isaac Lab · shared H1 robot asset"]
```

Run with the existing Isaac Sim interpreter from the project root:

```bash
/home/eddie/env_isaaclab/bin/python scripts/view_scene.py --appearance earth
/home/eddie/env_isaaclab/bin/python scripts/view_scene.py --appearance grand-canyon
```
