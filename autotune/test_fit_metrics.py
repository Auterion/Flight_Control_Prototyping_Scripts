"""Tests for the flown controller of a preset and the closed-loop replay fit.

Run with:  QT_QPA_PLATFORM=offscreen poetry run pytest test_fit_metrics.py
"""

import os
from types import SimpleNamespace

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


def test_gain_scale_multiplies_the_parameter_values(selection):
    select_preset(selection, "Rollrate", gain_scale=0.001)
    selection.loadSelection()

    scaled = {gain: 0.001 * value for gain, value in X500_ROLL_GAINS.items()}
    assert selection.flown_gains == pytest.approx(idealGains(scaled), rel=1e-5)


def test_gain_scale_is_loaded_with_the_preset(selection):
    select_preset(selection, "Rollrate", gain_scale=0.001)
    selection.loadSelection()

    assert selection.flown_gain_scale == 0.001


def test_loaded_gains_are_shown_in_parameter_units(selection, monkeypatch):
    select_preset(selection, "Rollrate", gain_scale=0.001)

    window = load_log(selection, monkeypatch)

    assert window.gain_scale == 0.001
    assert float(window.gain_edit["parallel"]["P"].text()) == pytest.approx(
        X500_ROLL_GAINS["P"], abs=1e-4
    )


def test_minimum_gains_are_in_parameter_units(selection, monkeypatch):
    select_preset(selection, "Rollrate", gain_scale=0.001)
    del selection.preset_candidates["Rollrate"]["gains"]

    window = load_log(selection, monkeypatch)

    assert float(window.gain_edit["parallel"]["P"].text()) == pytest.approx(
        autotune.kMinimumGains["P"]
    )


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
    monkeypatch.setattr(autotune, "DataSelectionWindow", lambda *args: selection)
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


def test_validation_range_loads_its_own_data_reference_and_gains(selection):
    select_preset(selection, "Rollrate")
    selection.validation_range = (103.0, 109.0)

    selection.loadSelection()

    validation = selection.validation
    assert validation.t[0] == pytest.approx(103.0, abs=0.1)
    assert validation.t[-1] == pytest.approx(109.0, abs=0.1)
    assert selection.t[0] == pytest.approx(76.5, abs=0.1)
    assert len(validation.u) == len(validation.y) == len(validation.t)
    assert len(validation.r) == len(validation.t)
    assert validation.flown_gains == pytest.approx(
        idealGains(X500_ROLL_GAINS), rel=1e-5
    )


def test_no_validation_range_loads_no_validation_data(selection):
    select_preset(selection, "Rollrate")
    selection.loadSelection()

    assert selection.validation is None


def test_model_is_judged_on_the_validation_data(selection, monkeypatch):
    select_preset(selection, "Rollrate")
    window = load_log(selection, monkeypatch)
    estimation_fit = window.lbl_fit.text()
    estimation_replay = window.lbl_closed_loop_fit.text()

    selection.validation_range = (103.0, 109.0)
    window = load_log(selection, monkeypatch)

    validation = window.validation
    assert window.dt == pytest.approx(np.mean(np.diff(validation.t)))
    assert window.lbl_fit.text() != estimation_fit
    assert window.lbl_closed_loop_fit.text() != estimation_replay
    assert window.output_curve.yData == pytest.approx(validation.y)
    assert len(window.model_curve.xData) == len(validation.t)
    assert window.io_plot.titleLabel.text.startswith("Validation data")


def test_parameter_search_ranks_on_the_validation_data(selection, monkeypatch):
    select_preset(selection, "Rollrate")
    selection.validation_range = (103.0, 109.0)
    window = load_log(selection, monkeypatch)

    settings = search_settings(window, monkeypatch)

    u_v, y_v = settings["validation"]
    assert u_v == pytest.approx(window.validation.u)
    assert y_v == pytest.approx(window.validation.y)


def test_validation_criterion_is_the_prediction_error_on_the_validation_data():
    _, _, delay, u, y = integrator_with_disturbance()
    u_v, y_v = integrator_with_disturbance(seed=1)[3:]
    order = (1, 0, delay)
    sys_id = autotune.SystemIdentification(*order, DT)
    sys_id.f_hp, sys_id.f_lp = 0.0, 1000.0
    sys_id.fit(u, y, method="RLS")

    criterion = autotune.search_criterion(
        u, y, DT, *order, 0.0, 1000.0, horizon=20, validation=(u_v, y_v)
    )

    assert criterion == pytest.approx(
        autotune.compute_prediction_mse(
            sys_id.getNum(), sys_id.getDen(), delay, u_v, y_v, 20
        )
    )


def test_preset_without_gains_loads_the_minimum_gains_in_its_form(
    selection, monkeypatch
):
    select_preset(selection, "Rollrate", gain_form="ideal")
    del selection.preset_candidates["Rollrate"]["gains"]

    window = load_log(selection, monkeypatch)

    assert window.gains == autotune.kMinimumGains
    assert window.gain_form_radio["ideal"].isChecked()
    assert window.lbl_closed_loop_fit.text() == "—"  # nothing to replay


def test_offset_removal_checkbox_drives_the_identification(selection, monkeypatch):
    select_preset(selection, "Rollrate")
    window = load_log(selection, monkeypatch)
    assert window.remove_offset_box.isChecked()

    used = []

    class RecordingIdentification(autotune.SystemIdentification):
        def fit(self, u, y, method="RLS"):
            used.append(self.remove_offset)
            return super().fit(u, y, method)

    monkeypatch.setattr(autotune, "SystemIdentification", RecordingIdentification)
    window.remove_offset_box.setChecked(False)
    window.runIdentification()

    assert used == [False]


def search_settings(window, monkeypatch):
    """Keyword arguments of the parameter search started from the window."""
    searches = []

    class StubWorker:
        def __init__(self, *args, **kwargs):
            searches.append(kwargs)
            self.progress = self.finished = SimpleNamespace(connect=lambda slot: None)

        def start(self):
            pass

    monkeypatch.setattr(autotune, "ParamSearchWorker", StubWorker)
    window.onFindParamsClicked()
    return searches[0]


def test_offset_removal_checkbox_drives_the_parameter_search(selection, monkeypatch):
    select_preset(selection, "Rollrate")
    window = load_log(selection, monkeypatch)
    window.remove_offset_box.setChecked(False)

    assert search_settings(window, monkeypatch)["remove_offset"] is False


def test_parameter_search_uses_the_prediction_horizon(selection, monkeypatch):
    select_preset(selection, "Rollrate")
    window = load_log(selection, monkeypatch)
    window.horizon_spinbox.setValue(0.05)

    settings = search_settings(window, monkeypatch)

    assert settings["horizon"] == round(0.05 / window.dt)
    assert "flown_controller" not in settings


def test_parameter_search_result_fits_in_the_model_order_spinboxes(
    selection, monkeypatch
):
    select_preset(selection, "Rollrate")
    window = load_log(selection, monkeypatch)
    largest = autotune.kMaxModelOrder
    monkeypatch.setattr(window, "onSysIdClicked", lambda: None)

    window.onParamSearchFinished(
        {"n_poles": largest, "n_zeros": largest, "delay": 0}, 0.0
    )

    assert window.line_edit_poles.value() == largest
    assert window.line_edit_zeros.value() == largest


def test_coefficient_table_of_a_large_model_scrolls(selection, monkeypatch):
    select_preset(selection, "Rollrate")
    window = load_log(selection, monkeypatch)
    small_height = window.t_coeffs.height()

    window.line_edit_poles.setValue(autotune.kMaxModelOrder)
    window.line_edit_zeros.setValue(autotune.kMaxModelOrder)
    window.show()

    table = window.t_coeffs
    assert table.rowCount() > autotune.kMaxCoefficientRowsShown
    assert small_height < table.height() < table.verticalHeader().length()
    assert table.verticalScrollBar().isVisible()


def test_parameter_search_ranks_a_worse_predicting_larger_model_lower(
    selection, monkeypatch
):
    select_preset(selection, "Rollrate")
    window = load_log(selection, monkeypatch)
    horizon = window.predictionHorizon()

    def criterion(n_poles, n_zeros, delay):
        return autotune.search_criterion(
            window.u,
            window.y,
            window.dt,
            n_poles,
            n_zeros,
            delay,
            0.0,
            30.0,
            horizon=horizon,
        )

    assert criterion(2, 2, 1) < criterion(6, 6, 3)


def test_prediction_horizon_updates_the_fit_without_a_new_identification(
    selection, monkeypatch
):
    select_preset(selection, "Rollrate")
    window = load_log(selection, monkeypatch)
    window.runIdentification()
    short_horizon_fit = window.lbl_fit.text()
    num = list(window.num)
    model_curve = window.model_curve.getData()[1].copy()

    window.horizon_spinbox.setValue(1.0)

    assert window.lbl_fit.text() != short_horizon_fit
    assert window.num == num
    # the model curve is the simulation, whatever the horizon
    assert window.model_curve.getData()[1] == pytest.approx(model_curve)


def integrator_with_disturbance(seed=0):
    """Rate of an integrating plant y(k) = y(k-1) + 0.05 u(k-3), under
    proportional control and an unmeasured slow disturbance torque."""
    rng = np.random.default_rng(seed)
    n = 1250
    reference = np.sign(np.sin(2 * np.pi * np.arange(n) * DT))
    disturbance = np.cumsum(rng.standard_normal(n)) * 0.003
    u = np.zeros(n)
    y = np.zeros(n)
    for k in range(n):
        if k >= 3:
            y[k] = y[k - 1] + 0.05 * (u[k - 3] + disturbance[k - 3])
        u[k] = 4.0 * (reference[k] - y[k])
    return [0.05], [1.0, -1.0], 3, u, y


def test_prediction_fit_of_the_true_model_is_not_spoiled_by_integrator_drift():
    num, den, delay, u, y = integrator_with_disturbance()

    prediction_fit = autotune.compute_prediction_fit(num, den, delay, u, y, 20)
    # 2 s of 6: close to a simulation, which drifts with the disturbance
    long_horizon_fit = autotune.compute_prediction_fit(num, den, delay, u, y, 400)

    assert prediction_fit > 90.0
    assert long_horizon_fit < 50.0


def test_prediction_fit_ranks_a_wrong_model_lower():
    num, den, delay, u, y = integrator_with_disturbance()

    true_fit = autotune.compute_prediction_fit(num, den, delay, u, y, 20)
    wrong_delay_fit = autotune.compute_prediction_fit(num, den, 0, u, y, 20)
    wrong_gain_fit = autotune.compute_prediction_fit([0.03], den, delay, u, y, 20)

    assert true_fit > wrong_delay_fit + 10.0
    assert true_fit > wrong_gain_fit + 10.0


@pytest.mark.parametrize("order", [(2, 2, 3), (4, 4, 3), (7, 7, 0)])
def test_predicted_squared_error_is_least_for_the_true_model(order):
    num, den, delay, u, y = integrator_with_disturbance()

    true_pse = autotune.compute_predicted_squared_error(num, den, delay, u, y, 20)
    larger_pse = autotune.search_criterion(u, y, DT, *order, 0.0, 1000.0, horizon=20)

    assert true_pse < larger_pse


def test_predicted_squared_error_penalizes_a_useless_parameter():
    num, den, delay, u, y = integrator_with_disturbance()
    # a zero leading coefficient: the same prediction, one parameter more
    padded_pse = autotune.compute_predicted_squared_error(
        [0.0] + num, den, delay - 1, u, y, 20
    )
    y_valid, _ = autotune.predictedOutput(num, den, delay, u, y, 20)

    assert padded_pse - autotune.compute_predicted_squared_error(
        num, den, delay, u, y, 20
    ) == pytest.approx(np.var(y_valid) / len(y_valid))
