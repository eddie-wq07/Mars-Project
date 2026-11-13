"""Catch missing LFS content before RTX renders its cyan error material."""

import pytest

from mars_project.assets import scene_paths


def test_rejects_lfs_pointer(tmp_path, monkeypatch):
    monkeypatch.setattr(scene_paths, "GRAND_CANYON_DIR", tmp_path)
    optimized = tmp_path / "optimized"
    optimized.mkdir()
    for name in ("scenery.usdc", "local_ground.usdc", "albedo.jpg"):
        (optimized / name).write_bytes(b"asset content")
    (optimized / "albedo.jpg").write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123\n")
    with pytest.raises(RuntimeError, match="Git LFS pointer"):
        scene_paths.validate_assets("grand-canyon")


def test_rejects_referenced_texture_pointer(tmp_path, monkeypatch):
    from pxr import Sdf, Usd

    monkeypatch.setattr(scene_paths, "EARTH_DIR", tmp_path)
    usd = tmp_path / "usd"
    usd.mkdir()
    stage = Usd.Stage.CreateNew(str(usd / "verdant_trail_earth.usdc"))
    stage.DefinePrim("/Scene").CreateAttribute("texture", Sdf.ValueTypeNames.Asset).Set("./sky.hdr")
    stage.GetRootLayer().Save()
    ground = Usd.Stage.CreateNew(str(usd / "earth_collision_ground.usda"))
    ground.GetRootLayer().Save()
    texture = usd / "sky.hdr"
    texture.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123\n")
    with pytest.raises(RuntimeError, match="Git LFS pointer"):
        scene_paths.validate_assets("earth")
    texture.write_bytes(b"hydrated texture")
    scene_paths.validate_assets("earth")
    texture.unlink()
    with pytest.raises(FileNotFoundError, match="Unresolved scene dependencies"):
        scene_paths.validate_assets("earth")
