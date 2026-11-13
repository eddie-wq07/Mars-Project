"""One fixed scene camera for preview and future recording."""


def create_camera(stage, cfg):
    """Use the scene's configured composition without camera animation."""
    from pxr import Gf, UsdGeom

    path = "/World/Cameras/Preview"
    camera = UsdGeom.Camera.Define(stage, path)
    view = cfg.sim.default_visualizer_cfg or cfg.viewer
    camera.CreateFocalLengthAttr(getattr(view, "focal_length", 28.0))
    camera.CreateHorizontalApertureAttr(36.0)
    camera.CreateVerticalApertureAttr(20.25)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.03, 10000.0))
    camera.CreateFStopAttr(0.0)
    matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*view.eye), Gf.Vec3d(*view.lookat), Gf.Vec3d(0, 0, 1)).GetInverse()
    UsdGeom.Xformable(camera).AddTransformOp().Set(matrix)
    return path
