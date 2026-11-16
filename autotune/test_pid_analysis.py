"""Tests of the plot of the step response estimated from logged data.

Run with:  QT_QPA_PLATFORM=offscreen poetry run pytest test_pid_analysis.py
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from pid_analyse_window import (  # noqa: E402
    PIDAnalyseWindow,
    compute_step_response_metrics,
    compute_step_responses,
)
from PyQt5.QtWidgets import QApplication  # noqa: E402
from scipy.signal import lfilter  # noqa: E402

DT = 0.005
TIME_CONSTANT = 0.03


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def window(qapp):
    return PIDAnalyseWindow()


def first_order_log():
    t = np.arange(0.0, 20.0, DT)
    rng = np.random.default_rng(0)
    u = np.repeat(rng.choice([-1.0, 1.0], size=len(t) // 40), 40)
    alpha = DT / (TIME_CONSTANT + DT)
    y = lfilter([alpha], [1, alpha - 1], u)
    return u, y, t


def is_empty(curve):
    return curve.xData is None or len(curve.xData) == 0


def first_order_step_response(time):
    alpha = DT / (TIME_CONSTANT + DT)
    return 1 - (1 - alpha) ** (np.round(time / DT) + 1)


def test_the_estimated_response_settles_at_one(window):
    metrics = window.generate_step_response(*first_order_log())

    time, mean = window.mean_response_curve.getData()
    assert time[0] == 0.0
    assert mean[-1] == pytest.approx(1.0, abs=0.05)
    assert metrics["steady_state_error"] == pytest.approx(0.0, abs=0.05)


def test_response_does_not_depend_on_the_units_of_the_signals():
    u, y, t = first_order_log()

    _, response, _ = compute_step_responses(u, y, t)
    _, response_deg, _ = compute_step_responses(np.degrees(u), np.degrees(y), t)

    assert response_deg == pytest.approx(response)


def test_quiet_parts_of_the_log_do_not_bias_the_response():
    u, y, t = first_order_log()
    is_quiet = (t < 8.0) | (t > 12.0)
    rng = np.random.default_rng(1)
    u = np.where(is_quiet, 0.0, u) + 1e-3 * rng.standard_normal(len(t))
    alpha = DT / (TIME_CONSTANT + DT)
    y = lfilter([alpha], [1, alpha - 1], u) + 0.02 * rng.standard_normal(len(t))

    time, response, window_responses = compute_step_responses(u, y, t)

    assert response == pytest.approx(first_order_step_response(time), abs=0.07)
    assert window_responses.mean(axis=0)[-1] == pytest.approx(1.0, abs=0.1)


def test_metrics_are_those_of_the_estimated_response(window):
    metrics = window.generate_step_response(*first_order_log())

    time, mean = window.mean_response_curve.getData()
    assert metrics == compute_step_response_metrics(time, mean)


def test_metrics_of_a_first_order_response():
    time = np.arange(0.0, 0.5, DT)

    metrics = compute_step_response_metrics(time, first_order_step_response(time))

    # 10% to 90% takes ln(9) time constants, 90% is reached after ln(10)
    assert metrics["rise_time (s)"] == pytest.approx(np.log(9) * TIME_CONSTANT, abs=DT)
    assert metrics["settling_time (s)"] == pytest.approx(
        np.log(10) * TIME_CONSTANT, abs=2 * DT
    )
    assert metrics["overshoot (%)"] == pytest.approx(0.0, abs=0.01)
    assert metrics["steady_state_error"] == pytest.approx(0.0, abs=1e-3)


def test_every_window_is_drawn_in_one_broken_curve(window):
    u, y, t = first_order_log()
    _, _, responses = compute_step_responses(u, y, t)

    window.generate_step_response(u, y, t)

    x, values = window.window_responses_curve.xData, window.window_responses_curve.yData
    assert np.count_nonzero(np.isnan(values)) == len(responses)
    assert len(x) == responses.size + len(responses)


def test_the_band_spans_one_standard_deviation(window):
    u, y, t = first_order_log()
    _, _, responses = compute_step_responses(u, y, t)

    window.generate_step_response(u, y, t)

    _, upper = window.std_upper_curve.getData()
    _, lower = window.std_lower_curve.getData()
    assert (upper - lower) / 2 == pytest.approx(responses.std(axis=0))


def test_metrics_are_listed(window):
    window.generate_step_response(*first_order_log())

    lines = window.metrics_text.text().splitlines()
    assert lines[0].startswith("rise_time (s): 0.0")
    assert "±" not in lines[0]
    assert any(line.startswith("overshoot (%):") for line in lines)


def test_constant_time_clears_the_plot(window):
    window.generate_step_response(*first_order_log())

    t = np.zeros(1000)
    assert window.generate_step_response(np.ones_like(t), np.ones_like(t), t) == {}
    assert is_empty(window.mean_response_curve)
    assert window.metrics_text.text() == ""


@pytest.mark.parametrize(
    "u, y, t",
    [
        pytest.param(*(signal[:150] for signal in first_order_log()), id="too short"),
        pytest.param(
            np.zeros(1000), np.ones(1000), np.arange(1000) * DT, id="no input"
        ),
    ],
)
def test_data_without_a_response_clear_the_plot(window, u, y, t):
    window.generate_step_response(*first_order_log())

    assert window.generate_step_response(u, y, t) == {}
    assert is_empty(window.mean_response_curve)
