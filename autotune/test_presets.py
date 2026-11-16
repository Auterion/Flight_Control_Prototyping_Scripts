"""Unit tests for preset persistence (presets.py).

Run with:  poetry run pytest test_presets.py
"""

import presets
import pytest


def _redirect(monkeypatch, tmp_path):
    """Point presets.py at a throwaway file so the real presets.yaml is safe."""
    path = tmp_path / "presets.yaml"
    monkeypatch.setattr(presets, "presets_file_path", lambda: str(path))
    return path


def test_load_seeds_defaults_when_missing(tmp_path, monkeypatch):
    path = _redirect(monkeypatch, tmp_path)
    assert not path.exists()

    loaded = presets.load_presets()

    assert path.exists()  # file is seeded on first run
    assert loaded == presets.DEFAULT_PRESETS


def test_round_trip_preserves_insertion_order(tmp_path, monkeypatch):
    _redirect(monkeypatch, tmp_path)
    data = {
        "Zulu": {"input": "z/in.0", "output": "z/out.0"},
        "Alpha": {"input": "a/in.0", "output": "a/out.0"},
    }

    presets.save_presets(data)
    loaded = presets.load_presets()

    assert loaded == data
    assert list(loaded.keys()) == ["Zulu", "Alpha"]  # not alphabetised


def test_legacy_keys_survive_round_trip(tmp_path, monkeypatch):
    _redirect(monkeypatch, tmp_path)
    data = {
        "Rollrate": {
            "input": "vehicle_torque_setpoint/xyz[0].0",
            "output": "vehicle_angular_velocity/xyz[0].0",
            "input_legacy": "actuator_controls_0/control[0].0",
        }
    }

    presets.save_presets(data)

    assert presets.load_presets() == data


def test_corrupt_file_falls_back_to_defaults(tmp_path, monkeypatch):
    path = _redirect(monkeypatch, tmp_path)
    path.write_text(":\n  - [ this is not valid yaml\n")

    loaded = presets.load_presets()

    assert loaded == presets.DEFAULT_PRESETS


def test_non_mapping_file_falls_back_to_defaults(tmp_path, monkeypatch):
    path = _redirect(monkeypatch, tmp_path)
    path.write_text("- just\n- a\n- list\n")

    loaded = presets.load_presets()

    assert loaded == presets.DEFAULT_PRESETS


@pytest.mark.parametrize(
    "name, axis", [("Rollrate(FW)", 0), ("Pitchrate(FW)", 1), ("Yawrate(FW)", 2)]
)
def test_fixed_wing_rate_presets_have_the_gain_compression_of_their_axis(name, axis):
    preset = presets.DEFAULT_PRESETS[name]

    assert preset["output"] == f"vehicle_angular_velocity/xyz[{axis}].0"
    assert set(preset["gains"]) == {"P", "I", "D", "FF"}
    assert preset["gain_compression"] == (
        f"gain_compression/compression_gains[{axis}].0"
    )
