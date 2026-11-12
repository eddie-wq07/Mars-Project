"""Camera creation, robot POV and eased orbit motion. No recording or simulator launch."""
import math


def create_cameras(stage, cfg, pov_offset):
    """Create selectable cinematic and body-tracked virtual cameras, not physical sensors."""
    from pxr import UsdGeom, Gf
    paths = {'robot': '/World/Cameras/RobotPOV', 'cinematic': '/World/Cameras/Cinematic'}
    for path in paths.values():
        cam = UsdGeom.Camera.Define(stage, path)
        cam.CreateFocalLengthAttr(28.)
        cam.CreateHorizontalApertureAttr(36.)
        cam.CreateVerticalApertureAttr(20.25)
        cam.CreateClippingRangeAttr(Gf.Vec2f(.03, 10000.))
        cam.CreateFocusDistanceAttr(5.)
        cam.CreateFStopAttr(0.)  # disable depth-of-field blur
        UsdGeom.Xformable(cam).AddTransformOp()
    view = cfg.sim.default_visualizer_cfg
    eye = view.eye if view else cfg.viewer.eye
    lookat = view.lookat if view else cfg.viewer.lookat
    matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*lookat), Gf.Vec3d(0, 0, 1)).GetInverse()
    UsdGeom.Xformable(stage.GetPrimAtPath(paths['cinematic'])).GetOrderedXformOps()[0].Set(matrix)
    return paths


def update_cinematic_camera(stage, camera_path, cfg, seconds, duration, degrees,
                            center=None, radius=3., height=.8):
    """Ease an orbit around the original composition's look-at point."""
    from pxr import Gf, UsdGeom
    view = cfg.sim.default_visualizer_cfg or cfg.viewer
    eye, target = Gf.Vec3d(*view.eye), Gf.Vec3d(*view.lookat)
    if center is not None:
        target = Gf.Vec3d(*center)
        azimuth = math.atan2(eye[1]-target[1], eye[0]-target[0])
        eye = target + Gf.Vec3d(radius*math.cos(azimuth), radius*math.sin(azimuth), height)
    progress = min(1., max(0., seconds / duration))
    eased = progress * progress * (3. - 2. * progress)
    angle = math.radians(degrees) * eased
    delta = eye - target
    rotated = Gf.Vec3d(delta[0]*math.cos(angle)-delta[1]*math.sin(angle),
                       delta[0]*math.sin(angle)+delta[1]*math.cos(angle), delta[2])
    matrix = Gf.Matrix4d().SetLookAt(target+rotated, target, Gf.Vec3d(0, 0, 1)).GetInverse()
    UsdGeom.Xformable(stage.GetPrimAtPath(camera_path)).GetOrderedXformOps()[0].Set(matrix)


def update_robot_camera(stage, robot, camera_path, offset):
    """Read the live body pose, so POV follows correctly even with Fabric enabled."""
    from pxr import Gf, UsdGeom
    names = robot.body_names
    if 'torso_link' not in names:
        raise RuntimeError(f'H1 torso_link not found; available bodies: {names}')
    index = names.index('torso_link')
    pose = robot.data.body_link_pose_w.torch[0, index].detach().cpu().tolist()
    # Isaac Lab 3 body poses use quaternion xyzw.
    body = Gf.Matrix4d().SetRotate(Gf.Quatd(pose[6], Gf.Vec3d(*pose[3:6])))
    body.SetTranslateOnly(Gf.Vec3d(*pose[:3]))
    local = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*offset), Gf.Vec3d(offset[0]+1., offset[1], offset[2]), Gf.Vec3d(0, 0, 1)).GetInverse()
    UsdGeom.Xformable(stage.GetPrimAtPath(camera_path)).GetOrderedXformOps()[0].Set(local * body)

