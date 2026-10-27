"""Tests for the flown controller of a preset and the closed-loop replay fit.

Run with:  QT_QPA_PLATFORM=offscreen poetry run pytest test_fit_metrics.py
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import presets  # noqa: E402
import pytest  # noqa: E402
from data_extractor import get_aos_settings  # noqa: E402
from data_selection_window import DataSelectionWindow  # noqa: E402
from loop_model import LoopModel, idealGains, parallelGains  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

import autotune  # noqa: E402

DT = 0.005
NUM = [0.0, 0.05, 0.03]
DEN = [1.0, -1.6, 0.65]
DELAYS = 2
GAINS = {"P": 0.8, "I": 0.3, "D": 0.02, "FF": 0.1}
UNSTABLE_GAINS = {"P": 1.5, "I": 0.5, "D": 0.08, "FF": 0.0}
# MC_ROLLRATE_* of the x500 log, never changed in flight
X500_ROLL_GAINS = {"P": 0.1430933, "I": 0.3379838, "D": 0.00027182, "FF": 0.0}


def logged_closed_loop(**pid_options):
    t = np.arange(0.0, 6.0, DT)
    r = np.where(np.sin(2 * np.pi * 0.8 * t) > 0, 1.0, -1.0)
    loop = LoopModel(NUM, DEN, DT, DELAYS, GAINS, **pid_options)
    return t, r, loop.simulate(t, r, np.zeros_like(t))


def test_parallel_gains_convert_to_ideal():
    ideal = idealGains({"P": 0.15, "I": 0.2, "D": 0.003, "FF": 0.05})

    assert ideal["P"] == pytest.approx(0.15)
    assert ideal["P"] * ideal["I"] == pytest.approx(0.2)
    assert ideal["P"] * ideal["D"] == pytest.approx(0.003)
    assert ideal["FF"] == pytest.approx(0.05)


def test_ideal_gains_are_kept():
    gains = {"P": 0.15, "I": 2.0, "D": 0.01, "FF": 0.05}
    assert idealGains(gains, "ideal") == gains


def test_parallel_gains_round_trip():
    gains = {"P": 0.15, "I": 0.2, "D": 0.003, "FF": 0.05}
    assert parallelGains(idealGains(gains)) == pytest.approx(gains)


def test_missing_gains_default_to_no_i_d_ff():
    assert idealGains({"P": 0.15}) == {"P": 0.15, "I": 0.0, "D": 0.0, "FF": 0.0}


def test_parallel_gains_without_p_have_no_ideal_form():
    assert idealGains({"P": 0.0, "I": 0.2}) is None


def test_replay_of_the_true_loop_is_a_perfect_fit():
    t, r, y = logged_closed_loop()
    fit = autotune.compute_closed_loop_fit(NUM, DEN, DT, DELAYS, GAINS, t, r, y)
    assert fit == pytest.approx(100.0, abs=1e-6)


def test_replay_uses_the_pid_options():
    t, r, y = logged_closed_loop(p_on_feedback=True)

    fit_with = autotune.compute_closed_loop_fit(
        NUM, DEN, DT, DELAYS, GAINS, t, r, y, p_on_feedback=True
    )
    fit_without = autotune.compute_closed_loop_fit(NUM, DEN, DT, DELAYS, GAINS, t, r, y)

    assert fit_with == pytest.approx(100.0, abs=1e-6)
    assert fit_without < fit_with


def test_replay_with_a_wrong_plant_gain_lowers_the_fit():
    t, r, y = logged_closed_loop()
    wrong_num = [0.5 * b for b in NUM]
    fit = autotune.compute_closed_loop_fit(wrong_num, DEN, DT, DELAYS, GAINS, t, r, y)
    assert fit < 90.0


def test_unstable_replay_has_no_fit():
    t, r, y = logged_closed_loop()
    assert not LoopModel(NUM, DEN, DT, DELAYS, UNSTABLE_GAINS).isStable()
    fit = autotune.compute_closed_loop_fit(
        NUM, DEN, DT, DELAYS, UNSTABLE_GAINS, t, r, y
    )
    assert fit is None


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def selection(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(
        presets, "presets_file_path", lambda: str(tmp_path / "presets.yaml")
    )
    log = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    window = DataSelectionWindow(os.path.join(log, "quadrotor_x500.ulg"))
    window.t_start, window.t_stop = 76.5, 83.5
    return window


def select_preset(selection, name, **changes):
    selection.preset_candidates[name].update(changes)
    selection.combo_preset.setCurrentIndex(selection.combo_preset.findText(name))


def test_rate_preset_loads_the_flown_controller(selection):
    select_preset(selection, "Rollrate")
    selection.loadSelection()

    assert len(selection.r) == len(selection.t)
    assert selection.flown_gains == pytest.approx(idealGains(X500_ROLL_GAINS), rel=1e-5)
    assert selection.flown_gain_form == "parallel"
    assert selection.flown_options == {"p_on_feedback": False, "negate_output": False}


def test_preset_gain_form_and_options_are_loaded(selection):
    select_preset(
        selection, "Rollrate", gain_form="ideal", pi_no_zero=True, negate_output=True
    )
    selection.loadSelection()

    assert selection.flown_gains == pytest.approx(
        idealGains(X500_ROLL_GAINS, "ideal"), rel=1e-5
    )
    assert selection.flown_gain_form == "ideal"
    assert selection.flown_options == {"p_on_feedback": True, "negate_output": True}


def test_gains_missing_from_the_log_are_not_loaded(selection):
    select_preset(selection, "Rollrate", gains={"P": "NOT_A_PARAM"})
    selection.loadSelection()

    assert selection.flown_gains is None
    assert selection.r is not None


def test_numeric_aos_settings_are_parameters():
    class Log:
        msg_info_multiple_dict = {
            "aos_settings": [
                [
                    '[{"name": "a.kp", "type": "float64", "value": 0.2}, {"name": "a.n",',
                    ' "type": "int64", "value": 3}, {"name": "a.on", "type": "bool", '
                    '"value": true}, {"name": "a.ip", "type": "string", "value": "x"}]',
                ]
            ]
        }

    assert get_aos_settings(Log()) == {"a.kp": 0.2, "a.n": 3}


def test_signals_changed_away_from_the_preset_load_no_controller(selection):
    select_preset(selection, "Rollrate")
    selection.combo_y.setCurrentIndex(
        selection.combo_y.findText("vehicle_angular_velocity/xyz[1].0")
    )
    selection.loadSelection()

    assert selection.r is None
    assert selection.flown_gains is None
    assert selection.flown_gain_form is None
    assert selection.flown_options is None


def load_log(selection, monkeypatch):
    selection.exec_ = lambda: selection.loadSelection() or True
    monkeypatch.setattr(autotune, "DataSelectionWindow", lambda file_name: selection)
    window = autotune.Window()
    window.gains = {"P": 0.5, "I": 1.0, "D": 0.01, "FF": 0.2}
    window.loadLog()
    return window


def test_loading_a_log_preloads_the_flown_pid(selection, monkeypatch):
    select_preset(selection, "Rollrate", pi_no_zero=True)

    window = load_log(selection, monkeypatch)

    assert window.gains == pytest.approx(idealGains(X500_ROLL_GAINS), rel=1e-5)
    # the PX4 parameters are in parallel form: that column is the editable one
    assert window.gain_form_radio["parallel"].isChecked()
    assert not window.gain_edit["parallel"]["I"].isReadOnly()
    assert window.gain_edit["ideal"]["I"].isReadOnly()
    assert float(window.gain_edit["parallel"]["I"].text()) == pytest.approx(
        X500_ROLL_GAINS["I"], abs=1e-4
    )
    assert window.pid_no_zero_box.isChecked()
    assert not window.negate_control_box.isChecked()
    assert window.lbl_closed_loop_fit.text().endswith("%")


def test_preset_without_gains_loads_the_minimum_gains_in_its_form(
    selection, monkeypatch
):
    select_preset(selection, "Rollrate", gain_form="ideal")
    del selection.preset_candidates["Rollrate"]["gains"]

    window = load_log(selection, monkeypatch)

    assert window.gains == autotune.kMinimumGains
    assert window.gain_form_radio["ideal"].isChecked()
    assert window.lbl_closed_loop_fit.text() == "—"  # nothing to replay
