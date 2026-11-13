"""Launch the H1 Earth or Grand Canyon scene.

Run with an Isaac Sim Python environment:
    python scripts/view_scene.py --appearance earth
    python scripts/view_scene.py --appearance grand-canyon
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mars_project.scene_runner import main

if __name__ == "__main__":
    main()
