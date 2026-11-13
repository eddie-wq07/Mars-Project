"""Regression coverage for stale/default exposure on a newly selected view."""

from pxr import Sdf, Usd, UsdGeom, UsdRender

from mars_project.scenes.cameras import author_view_settings


def test_reconfiguring_a_view_clears_stale_exposure_without_changing_lens():
    stage = Usd.Stage.CreateInMemory()
    camera = UsdGeom.Camera.Define(stage, "/Camera")
    camera.CreateFocalLengthAttr(22.0)
    prim = camera.GetPrim()
    prim.CreateAttribute("exposure:iso", Sdf.ValueTypeNames.Float).Set(100.0)
    prim.CreateAttribute("omni:rtx:autoExposure:enabled", Sdf.ValueTypeNames.Bool).Set(True)
    product = UsdRender.Product.Define(stage, "/Render/Product").GetPrim()
    product.CreateAttribute("omni:rtx:lights:dome:samplingStrategy", Sdf.ValueTypeNames.Token).Set(
        "environmentMappedImageBasedLighting"
    )
    for _ in range(2):
        author_view_settings(stage, "/Camera", "/Render/Product")
    assert prim.GetAttribute("exposure:iso").Get() == 0.0
    assert prim.GetAttribute("omni:rtx:autoExposure:enabled").Get() is False
    assert camera.GetFocalLengthAttr().Get() == 22.0
    assert product.GetAttribute("omni:rtx:lights:dome:samplingStrategy").Get() == "imageBasedLighting"
    assert product.GetAttribute("omni:rtx:rendermode").Get() == "Real-Time Path-Tracing"
    assert "OmniRtxSettingsCommonAdvancedAPI_1" in product.GetMetadata("apiSchemas").GetAppliedItems()
