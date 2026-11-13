"""Record simulator frames as a QuickTime-compatible H.264 MP4."""

import math
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from mars_project.screenshots import capture_frame, frame_robot, screenshot_path


def record_video(
    sim,
    robot,
    action_step,
    steps,
    step_dt,
    label,
    *,
    requested=False,
    camera_motion="fixed",
    camera_target=(0.0, 0.0, 0.95),
):
    import imageio_ffmpeg
    from omni.kit.viewport.utility import get_active_viewport

    from isaaclab.sim.utils.stage import get_current_stage

    fps = 25
    stride = round(1.0 / (fps * step_dt))
    if stride < 1 or abs(stride * step_dt * fps - 1.0) > 1e-6 or steps % stride:
        raise ValueError("Recording requires control steps divisible into exact 25-fps frames")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    filename = f"{label}-orbit" if camera_motion == "orbit" else label
    output = screenshot_path(filename, subfolder=label, requested=requested, suffix=".mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("Recording requires a Kit viewport")
    viewport.resolution = (1920, 1080)
    stage = get_current_stage()
    frame_robot(stage, viewport.camera_path, robot)
    orbit = None
    if camera_motion == "orbit":
        # Let the policy settle for one unrecorded second before the six-second shot.
        for _ in range(round(1.0 / step_dt)):
            action_step()
        midpoint = frame_robot(stage, viewport.camera_path, robot)
        camera_target = (camera_target[0], camera_target[1], midpoint[2])
        orbit = OrbitCamera(stage, viewport.camera_path, camera_target)
        orbit.update(0.0)
    for _ in range(60):
        sim.render()
    with TemporaryDirectory(prefix="mars-video-") as temporary:
        frames = Path(temporary)
        for index in range(steps // stride):
            for _ in range(stride):
                if not sim.is_running():
                    raise RuntimeError("Simulation closed before recording finished")
                action_step()
            if orbit is not None:
                orbit.update(index / max(1, steps // stride - 1))
                # Render the moved camera before scheduling its frame capture.
                sim.render()
            capture_frame(sim, viewport, frames / f"{index:06d}.png")
            if (index + 1) % fps == 0:
                print(f"VIDEO_PROGRESS seconds={(index + 1) / fps:.0f}", flush=True)
        subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-v",
                "error",
                "-n",
                "-framerate",
                str(fps),
                "-i",
                str(frames / "%06d.png"),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output),
            ],
            check=True,
        )
    print(f"VIDEO_SAVED {output} frames={steps // stride} fps={fps} seconds={steps * step_dt:.3f}", flush=True)
    return output


class OrbitCamera:
    """A repeatable world-space cinematic arc independent of robot pose or motion."""

    def __init__(self, stage, camera_path, camera_target):
        from pxr import UsdGeom

        self.camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
        self.transform = UsdGeom.Xformable(self.camera).GetOrderedXformOps()[0]
        self.center = tuple(camera_target)
        original_eye = self.transform.Get().ExtractTranslation()
        self.start_angle = math.atan2(original_eye[1] - self.center[1], original_eye[0] - self.center[0])
        self.camera.GetFocalLengthAttr().Set(22.0)

    def update(self, progress):
        from pxr import Gf

        eased = progress * progress * (3.0 - 2.0 * progress)
        angle = self.start_angle + math.radians(-20.0 + 80.0 * eased)
        target = Gf.Vec3d(*self.center)
        eye = target + Gf.Vec3d(2.6 * math.cos(angle), 2.6 * math.sin(angle), 0.0)
        self.transform.Set(Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse())
