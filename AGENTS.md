# Project output conventions

- Store all run, simulator, test and diagnostic `.log` files in `outputs/log/`. Create the directory before redirecting output. Do not write logs directly under `outputs/`.
- Store standalone lighting, sky and render-comparison PNG assets in `outputs/lighting/`, with descriptive names. Do not put PNG files directly under `outputs/`.
- Use the capture pipeline for simulation screenshots and videos: `outputs/recordings/MM-DD-YYYY/`. Captures not explicitly requested by the user belong in that date's `misc/` folder, including diagnostic pairs and videos. This rule takes precedence over the general lighting-image folder for pipeline captures.
- Use `--capture-requested` / `--screenshot-requested` (or `requested=True` in Python) only for screenshots explicitly requested by the user. Preserve any requested named subfolder such as `earth-test/`.
- Preserve existing outputs; use a unique name rather than overwriting. Keep README output examples consistent with these conventions.
