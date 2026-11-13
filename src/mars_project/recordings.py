"""Record simulator frames as a QuickTime-compatible H.264 MP4."""

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from mars_project.screenshots import capture_frame, frame_robot, screenshot_path


def record_video(sim, robot, action_step, steps, step_dt, label, *, requested=False):
    import imageio_ffmpeg
    from omni.kit.viewport.utility import get_active_viewport

    from isaaclab.sim.utils.stage import get_current_stage

    fps = 25
    stride = round(1.0 / (fps * step_dt))
    if stride < 1 or abs(stride * step_dt * fps - 1.0) > 1e-6 or steps % stride:
        raise ValueError("Recording requires control steps divisible into exact 25-fps frames")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    output = screenshot_path(label, subfolder=label, requested=requested, suffix=".mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("Recording requires a Kit viewport")
    viewport.resolution = (1920, 1080)
    frame_robot(get_current_stage(), viewport.camera_path, robot)
    for _ in range(60):
        sim.render()
    with TemporaryDirectory(prefix="mars-video-") as temporary:
        frames = Path(temporary)
        for index in range(steps // stride):
            for _ in range(stride):
                if not sim.is_running():
                    raise RuntimeError("Simulation closed before recording finished")
                action_step()
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
