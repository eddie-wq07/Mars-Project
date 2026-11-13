"""Capture the rendered Kit viewport without desktop UI or overlays."""

import asyncio
import time
from datetime import datetime
from pathlib import Path


def capture_screenshot(sim, output: Path | None = None):
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

    if output is None:
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S-%f%z")
        output = Path(__file__).resolve().parents[2] / "outputs" / "screenshots" / f"render_{stamp}.png"
    output = output.expanduser().resolve()
    if output.suffix.lower() != ".png":
        raise ValueError("Screenshots must use .png")
    output.parent.mkdir(parents=True, exist_ok=True)
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("Screenshot capture requires a Kit viewport")
    viewport.resolution = (1920, 1080)
    # Render without advancing physics while textures and lighting settle.
    for _ in range(60):
        sim.render()
    capture = capture_viewport_to_file(viewport, str(output))
    pending = asyncio.ensure_future(capture.wait_for_result())
    deadline = time.monotonic() + 60
    while not pending.done():
        if time.monotonic() > deadline:
            pending.cancel()
            raise TimeoutError("Viewport capture did not finish within 60 seconds")
        sim.render()
    pending.result()
    import omni.kit.renderer_capture

    writer = omni.kit.renderer_capture.acquire_renderer_capture_interface()
    for _ in range(3):
        writer.wait_async_capture()
        sim.render()
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Capture produced no image: {output}")
    print(f"SCREENSHOT_SAVED {output}", flush=True)
