"""Local scenery paths; H1 continues to use Isaac Lab's shared robot asset."""

from pathlib import Path

from mars_project.assets import MARS_PROJECT_ASSETS_DIR

EARTH_DIR = MARS_PROJECT_ASSETS_DIR / "polyhaven" / "verdant_trail"
GRAND_CANYON_DIR = MARS_PROJECT_ASSETS_DIR / "marslab" / "grand_canyon"


def validate_assets(appearance: str) -> None:
    """Fail before simulator startup for missing files or unhydrated Git LFS assets."""
    if appearance not in {"earth", "grand-canyon"}:
        raise ValueError(f"Unknown appearance: {appearance}")
    required = (
        (EARTH_DIR / "usd" / "verdant_trail_earth.usdc", EARTH_DIR / "usd" / "earth_collision_ground.usda")
        if appearance == "earth"
        else tuple(
            GRAND_CANYON_DIR / "optimized" / name
            for name in (
                "scenery.usdc",
                "local_ground.usdc",
                "albedo.jpg",
                "rock_normal.png",
                "roughness.png",
                "sky.png",
                "local_ground.npz",
            )
        )
    )
    for path in required:
        _validate_file(path)

    # USD materials reference textures that are not visible in Python imports.
    from pxr import UsdUtils

    for path in required:
        if path.suffix not in {".usd", ".usda", ".usdc"}:
            continue
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(path))
        if unresolved:
            raise FileNotFoundError(f"Unresolved scene dependencies: {unresolved}")
        for dependency in [*(layer.realPath for layer in layers), *assets]:
            _validate_file(Path(dependency))


def _validate_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Preview asset missing: {path}")
    with path.open("rb") as asset:
        if asset.read(64).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(
                f"Preview asset is a Git LFS pointer, not downloaded content: {path}. "
                "Run `git lfs pull` (or `git lfs checkout` if the objects are already cached)."
            )
