"""Local scenery paths; H1 continues to use Isaac Lab's shared robot asset."""

from mars_project.assets import MARS_PROJECT_ASSETS_DIR

EARTH_DIR = MARS_PROJECT_ASSETS_DIR / "polyhaven" / "verdant_trail"
GRAND_CANYON_DIR = MARS_PROJECT_ASSETS_DIR / "marslab" / "grand_canyon"


def validate_assets(appearance: str) -> None:
    """Fail before simulator startup if required scenery files are missing."""
    required = (
        (EARTH_DIR / "usd" / "verdant_trail_earth.usdc", EARTH_DIR / "usd" / "earth_collision_ground.usda")
        if appearance == "earth"
        else tuple(GRAND_CANYON_DIR / "optimized" / name for name in (
            "scenery.usdc", "local_ground.usdc", "albedo.jpg", "rock_normal.png", "roughness.png", "sky.png"
        ))
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Preview asset missing: {path}")
