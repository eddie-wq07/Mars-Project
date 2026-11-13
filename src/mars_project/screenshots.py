"""Capture the rendered Kit viewport without desktop UI or overlays."""

import asyncio
import re
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def screenshot_path(
    label: str,
    taken_at: datetime | None = None,
    subfolder: str | None = None,
    *,
    requested: bool = False,
    suffix: str = ".png",
) -> Path:
    """Route unsolicited captures to date/misc; preserve requested destinations and avoid overwrites."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise ValueError("Screenshot label must contain only letters, numbers, hyphens or underscores")
    if suffix not in (".png", ".mp4"):
        raise ValueError("Capture suffix must be .png or .mp4")
    taken_at = taken_at or datetime.now(ZoneInfo("America/Vancouver"))
    hour = taken_at.hour % 12 or 12
    period = "am" if taken_at.hour < 12 else "pm"
    stamp = f"{hour}:{taken_at:%M}{period}"
    folder = Path(__file__).resolve().parents[2] / "outputs" / "recordings" / f"{taken_at:%m-%d-%Y}"
    if not requested:
        subfolder = "misc"
    if subfolder:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", subfolder):
            raise ValueError("Screenshot subfolder must be a single folder name")
        folder = folder / subfolder
    output = folder / f"{stamp}_{label}{suffix}"
    sequence = 2
    while output.exists():
        output = folder / f"{stamp}_{label}-{sequence}{suffix}"
        sequence += 1
    return output


def frame_robot(stage, camera_path, robot):
    """Center live H1 body positions and fit roughly 40% of image height."""
    from pxr import Gf, UsdGeom

    camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
    xform = UsdGeom.Xformable(camera)
    original = xform.GetLocalTransformation()
    points = robot.data.body_link_pose_w.torch[0, :, :3].detach().cpu().tolist()
    low = [min(p[i] for p in points) - 0.15 for i in range(3)]
    high = [max(p[i] for p in points) + 0.15 for i in range(3)]
    center = Gf.Vec3d(*[(a + b) / 2 for a, b in zip(low, high, strict=True)])
    eye = original.ExtractTranslation()
    # Keep the scene's viewing direction, but aim at the robot's current position.
    matrix = Gf.Matrix4d().SetLookAt(eye, center, Gf.Vec3d(0, 0, 1)).GetInverse()
    view = matrix.GetInverse()
    corners = [
        view.Transform(Gf.Vec3d(x, y, z))
        for x in (low[0], high[0])
        for y in (low[1], high[1])
        for z in (low[2], high[2])
    ]
    if any(p[2] >= 0 for p in corners):
        raise RuntimeError("Robot overlaps screenshot camera; reposition the scene camera")
    span = max(p[1] / -p[2] for p in corners) - min(p[1] / -p[2] for p in corners)
    camera.GetFocalLengthAttr().Set(0.40 * camera.GetVerticalApertureAttr().Get() / span)
    xform.GetOrderedXformOps()[0].Set(matrix)


def capture_screenshot(sim, robot, label: str = "render", *, subfolder=None, reframe=True, requested=False):
    from omni.kit.viewport.utility import get_active_viewport

    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("Screenshot capture requires a Kit viewport")
    viewport.resolution = (1920, 1080)
    from isaaclab.sim.utils.stage import get_current_stage

    if reframe:
        frame_robot(get_current_stage(), viewport.camera_path, robot)
    # Render without advancing physics while textures and lighting settle.
    for _ in range(60):
        sim.render()
    output = screenshot_path(label, subfolder=subfolder, requested=requested)
    output.parent.mkdir(parents=True, exist_ok=True)
    capture_frame(sim, viewport, output)
    print(f"SCREENSHOT_SAVED {output}", flush=True)


def capture_frame(sim, viewport, output):
    """Write one viewport frame, rendering as needed without advancing physics."""
    from omni.kit.viewport.utility import capture_viewport_to_file

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
