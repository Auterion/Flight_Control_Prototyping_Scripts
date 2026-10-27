"""Tests of the plot of the step response estimated from logged data.

Run with:  QT_QPA_PLATFORM=offscreen poetry run pytest test_pid_analysis.py
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from pid_analyse_window import PIDAnalyseWindow, compute_step_responses  # noqa: E402
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


def test_the_estimated_response_settles_at_one(window):
    metrics = window.generate_step_response(*first_order_log())

    time, mean = window.mean_response_curve.getData()
    assert time[0] == 0.0
    assert mean[-1] == pytest.approx(1.0, abs=0.05)
    assert metrics["steady_state_error"][0] == pytest.approx(0.0, abs=0.05)


def test_every_window_is_drawn_in_one_broken_curve(window):
    u, y, t = first_order_log()
    _, responses = compute_step_responses(u, y, t)

    window.generate_step_response(u, y, t)

    x, values = window.window_responses_curve.xData, window.window_responses_curve.yData
    assert np.count_nonzero(np.isnan(values)) == len(responses)
    assert len(x) == responses.size + len(responses)


def test_the_band_spans_one_standard_deviation(window):
    u, y, t = first_order_log()
    _, responses = compute_step_responses(u, y, t)

    window.generate_step_response(u, y, t)

    _, upper = window.std_upper_curve.getData()
    _, lower = window.std_lower_curve.getData()
    assert (upper - lower) / 2 == pytest.approx(responses.std(axis=0))


def test_metrics_are_listed(window):
    window.generate_step_response(*first_order_log())

    lines = window.metrics_text.text().splitlines()
    assert lines[0].startswith("rise_time (s):")
    assert any(line.startswith("overshoot (%):") for line in lines)


def test_constant_time_clears_the_plot(window):
    window.generate_step_response(*first_order_log())

    t = np.zeros(1000)
    assert window.generate_step_response(np.ones_like(t), np.ones_like(t), t) == {}
    assert window.mean_response_curve.xData is None or (
        len(window.mean_response_curve.xData) == 0
    )
    assert window.metrics_text.text() == ""
