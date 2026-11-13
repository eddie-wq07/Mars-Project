"""Requested experiments are grouped; diagnostic captures always stay in misc."""

from datetime import datetime

from mars_project.screenshots import screenshot_path


def test_requested_groups_and_diagnostic_routing():
    date = datetime(2030, 1, 2, 11, 5)
    for group in ("earth", "mars"):
        path = screenshot_path(f"{group}-1-orbit", date, f"{group}-1", requested=True, suffix=".mp4")
        assert path.parts[-4:] == ("01-02-2030", group, f"{group}-1", f"11:05am_{group}-1-orbit.mp4")
        diagnostic = screenshot_path(f"{group}-check", date, f"{group}-check", suffix=".mp4")
        assert diagnostic.parts[-3:] == ("01-02-2030", "misc", f"11:05am_{group}-check.mp4")
