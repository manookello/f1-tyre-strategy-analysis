import pandas as pd

from f1tyres.openf1 import _track_status_by_lap


def test_track_status_from_race_control_messages():
    rc = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01T10:00:00Z"] * 6),
            "lap_number": [10, 13, 20, 21, 30, 5],
            "message": [
                "SAFETY CAR DEPLOYED",
                "SAFETY CAR IN THIS LAP",
                "VIRTUAL SAFETY CAR DEPLOYED",
                "VIRTUAL SAFETY CAR ENDING",
                "RED FLAG",
                "PIT EXIT CLOSED",
            ],
            "flag": [None, None, None, None, "RED", "RED"],
        }
    )
    status = _track_status_by_lap(rc, lambda ts: 1, total_laps=40)
    assert status[9] == "1"
    assert all(status[lap] == "4" for lap in range(10, 14))
    assert status[14] == "1"
    assert status[20] == "6" and status[21] == "6"
    assert "5" in status[30]
    assert status[5] == "1"  # a pit-exit red light is not a red flag


def test_track_status_all_green_without_messages():
    status = _track_status_by_lap(pd.DataFrame(), lambda ts: 1, total_laps=5)
    assert set(status.values()) == {"1"}
