"""One fixed scene camera for preview and future recording."""

# Scene-referred HDRs at Blender exposure 0. RTX uses ISO=0 to bypass its
# physical-camera multiplier (same neutral preset as installed isaaclab_ppisp's
# rtx_camera_overrides.py). This is RTX-specific, not a physical ISO or the
# UsdGeomCamera.ComputeLinearExposureScale formula. ACES remains enabled.
# Exposure controls do not change motion blur or depth of field.
CAMERA_EXPOSURE = {
    "exposure": 0.0,
    "exposure:iso": 0.0,
    "exposure:time": 1.0,
    "exposure:fStop": 1.0,
    "exposure:responsivity": 1.0,
}

VIEW_RENDER_SETTINGS = {
    "/rtx/post/tonemap/op": 6,  # ACES in the installed RTX settings schema
    "/rtx/post/tonemap/filmIso": CAMERA_EXPOSURE["exposure:iso"],
    "/rtx/post/tonemap/exposureTime": CAMERA_EXPOSURE["exposure:time"],
    "/rtx/post/tonemap/fNumber": CAMERA_EXPOSURE["exposure:fStop"],
    "/rtx/post/tonemap/responsivity": CAMERA_EXPOSURE["exposure:responsivity"],
    "/rtx/post/tonemap/colorMode": 0,  # linear sRGB inputs
    "/rtx/post/tonemap/whitepoint": (1.0, 1.0, 1.0),
    "/rtx/post/histogram/enabled": False,
    "/rtx/post/histogram/filterType": 0,  # median
    "/rtx/post/histogram/whiteScale": 1.0,
    "/rtx/post/histogram/tau": 1.0,
    "/rtx/post/histogram/useExposureClamping": False,
    "/rtx/post/tonemap/ocio/enabled": False,
}


def author_view_settings(stage, camera_path, render_product_path, render_mode="Real-Time Path-Tracing"):
    """Author effective per-view settings; also testable on an in-memory USD stage."""
    from pxr import Sdf, UsdGeom, UsdRender

    camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
    product = UsdRender.Product(stage.GetPrimAtPath(render_product_path))
    if not camera or not product:
        raise RuntimeError("Rendering requires an existing viewport camera and render product")
    # Kit's USD watcher requires these APIs to route per-view settings to RTX.
    # AddAppliedSchema also works in offline USD without loading Kit plugins.
    for schema in ["OmniRtxCameraExposureAPI_1", "OmniRtxCameraAutoExposureAPI_1"]:
        camera.GetPrim().AddAppliedSchema(schema)
    for schema in ["OmniRtxSettingsCommonAPI_1", "OmniRtxSettingsCommonAdvancedAPI_1", "OmniRtxPostTonemapAPI_1"]:
        product.GetPrim().AddAppliedSchema(schema)
    for name, value in CAMERA_EXPOSURE.items():
        camera.GetPrim().CreateAttribute(name, Sdf.ValueTypeNames.Float, custom=False).Set(value)
    camera.GetPrim().CreateAttribute("omni:rtx:autoExposure:enabled", Sdf.ValueTypeNames.Bool, custom=False).Set(False)
    for name, value in {
        "omni:rtx:background:source:type": "domeLight",
        "omni:rtx:lights:visibilityOverride": "perLight",
        # Author on the render product too: legacy global settings alone do not
        # describe the effective view after a viewport/camera change.
        "omni:rtx:lights:dome:samplingStrategy": "imageBasedLighting",
        "omni:rtx:rendermode": render_mode,
        "omni:rtx:post:tonemap:operator": "acesApproximation",
        "omni:rtx:post:tonemap:colorMode": "sRGBLinear",
    }.items():
        product.GetPrim().CreateAttribute(name, Sdf.ValueTypeNames.Token).Set(value)


def apply_view_settings(stage, camera_path, render_product_path):
    """Apply identical exposure/color processing to Earth and Mars after camera selection."""
    import carb.settings

    settings = carb.settings.get_settings()
    for name, value in VIEW_RENDER_SETTINGS.items():
        settings.set(name, value)
    author_view_settings(stage, camera_path, render_product_path)


def log_view_settings(stage, viewport):
    """Keep the selected view's authored state in each run's diagnostic log."""
    import json

    camera = stage.GetPrimAtPath(viewport.camera_path)
    product = stage.GetPrimAtPath(viewport.render_product_path)
    print(
        "RENDER_VIEW "
        + json.dumps(
            {
                "camera": str(viewport.camera_path),
                "render_product": str(viewport.render_product_path),
                "exposure": {name: camera.GetAttribute(name).Get() for name in CAMERA_EXPOSURE},
                "auto_exposure": camera.GetAttribute("omni:rtx:autoExposure:enabled").Get(),
                "mode": product.GetAttribute("omni:rtx:rendermode").Get(),
                "dome_sampling": product.GetAttribute("omni:rtx:lights:dome:samplingStrategy").Get(),
                "tonemap": product.GetAttribute("omni:rtx:post:tonemap:operator").Get(),
            },
            sort_keys=True,
        ),
        flush=True,
    )


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
