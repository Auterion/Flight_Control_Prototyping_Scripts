"""Tests of the data shown by the pyqtgraph plots of the main window.

Run with:  QT_QPA_PLATFORM=offscreen poetry run pytest test_plots.py
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import plot_style as ps  # noqa: E402
import pytest  # noqa: E402
from loop_model import LoopModel  # noqa: E402
from PyQt5.QtCore import QPointF  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402
from system_identification import arx_transfer_function  # noqa: E402

import autotune  # noqa: E402

DT = 0.005
NUM = [0.0, 0.05, 0.03]
DEN = [1.0, -1.6, 0.65]
DELAYS = 2
GAINS = {"P": 0.8, "I": 0.3, "D": 0.02, "FF": 0.1}
UNSTABLE_GAINS = {"P": 1.5, "I": 0.5, "D": 0.08, "FF": 0.0}


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def window(qapp):
    # Window reuses pyplot figure 2, which still points to the Qt canvas of
    # any previous window
    plt.close("all")
    window = autotune.Window()
    window.num, window.den, window.dt = NUM, DEN, DT
    window.sys_id_delays = DELAYS
    window.Gz = arx_transfer_function(NUM, DEN, DT)
    window.is_system_identified = True
    window.gains = dict(GAINS)
    window.plotPolesZeros()
    window.updateClosedLoop()
    return window


def loop_model(window):
    return LoopModel(NUM, DEN, DT, DELAYS, window.gains)


def change_gains(window, **gains):
    window.gains.update(gains)
    window.updateClosedLoop()


def test_step_response_follows_the_gains(window):
    change_gains(window, P=0.5)

    t, y = window.step_response_curve.getData()
    disturbance = np.where(
        t >= window.kDisturbanceTime, window.disturbance_amplitude, 0.0
    )
    expected = loop_model(window).simulate(t, np.ones_like(t), disturbance)
    assert y == pytest.approx(expected)


def test_poles_and_zeros_are_those_of_the_model(window):
    poles_x, poles_y = window.pole_markers.getData()
    expected = window.Gz.poles()
    assert sorted(poles_x + 1j * poles_y, key=np.angle) == pytest.approx(
        sorted(expected, key=np.angle)
    )


def test_bode_plots_the_loop_gain(window):
    curve = window.bode_mag_curves["open"]
    f, mag_db = curve.xData, curve.yData

    assert len(f) == 500
    w = 2 * np.pi * f
    response = loop_model(window).loop_gain(np.exp(1j * w * DT))
    assert mag_db == pytest.approx(20 * np.log10(np.abs(response)), abs=1e-6)


def test_bode_margins_are_colored_by_threshold(window):
    lines = window.bode_margin_text.text().splitlines()
    assert lines[0].startswith("Gain margin:")
    assert lines[1].startswith("Phase margin:")

    html = window.bode_margin_text.text_item.toHtml()
    assert ps.GOOD_TEXT in html or ps.CRITICAL in html


def test_unstable_loop_replaces_the_bode_margins(window):
    change_gains(window, **UNSTABLE_GAINS)

    assert window.bode_margin_text.text() == window.kUnstableLoopText
    assert ps.CRITICAL in window.bode_margin_text.text_item.toHtml()


def test_overshoot_limit_follows_the_step_info(window):
    window.step_info_spinbox["overshoot"].setValue(25.0)

    _, y_limit = window.overshoot_limit_curve.getData()
    assert y_limit == pytest.approx([1.25, 1.25])


def test_zoom_survives_a_gain_change(window):
    view_box = window.bode_mag_plot.getViewBox()
    view_box.setRange(xRange=(0.0, 1.0), yRange=(-5.0, 5.0), padding=0)

    change_gains(window, P=0.6)

    (x_min, x_max), (y_min, y_max) = view_box.viewRange()
    assert (x_min, x_max) == pytest.approx((0.0, 1.0))
    assert (y_min, y_max) == pytest.approx((-5.0, 5.0))


def test_reset_restores_the_default_range(window):
    view_box = window.bode_mag_plot.getViewBox()
    view_box.setRange(xRange=(0.0, 1.0), yRange=(-5.0, 5.0), padding=0)

    view_box.resetRange()

    f_nyquist = 1 / (2 * DT)
    (x_min, x_max), (y_min, y_max) = view_box.viewRange()
    assert (x_min, x_max) == pytest.approx(
        (np.log10(0.1 / (2 * np.pi)), np.log10(f_nyquist))
    )
    assert (y_min, y_max) == pytest.approx((-20.0, 20.0))


def test_bode_phase_shares_the_frequency_axis(window):
    mag_view = window.bode_mag_plot.getViewBox()
    phase_view = window.bode_phase_plot.getViewBox()
    mag_view.setXRange(0.0, 1.0, padding=0)

    # Linked views align on screen: a frequency sits at the same pixel column,
    # within the rounding of the axis widths to whole pixels
    for log_f in (0.0, 0.5, 1.0):
        scene_x = mag_view.mapViewToScene(QPointF(log_f, 0)).x()
        phase_log_f = phase_view.mapSceneToView(QPointF(scene_x, 0)).x()
        assert phase_log_f == pytest.approx(log_f, abs=0.01)


def test_reset_clears_the_plots(window):
    window.reset()

    for curve in window.generalCurves():
        assert curve.xData is None or len(curve.xData) == 0


def test_hover_readout_shows_the_frequency_in_hz(window):
    window.resize(1200, 900)
    window.show()
    view_box = window.bode_mag_plot.getViewBox()
    view_box.setRange(xRange=(0.0, 2.0), yRange=(-10.0, 10.0), padding=0)
    readout = window.hover_readouts[1]

    scene_pos = view_box.mapViewToScene(QPointF(1.0, 5.0))
    readout.onMouseMoved(scene_pos)

    assert readout.label.isVisible()
    assert readout.label.text() == "f = 10 Hz, 5.0 dB"


@pytest.mark.parametrize("plot_name", ["pz_plot"])
def test_aspect_locked_axis_does_not_shrink_back(qapp, plot_name):
    window = autotune.Window()
    window.resize(1400, 900)
    window.show()
    plot = getattr(window, plot_name)
    view = plot.getViewBox().scene().views()[0]
    window.plot_tabs.setCurrentWidget(view)
    axis = plot.getAxis("left")

    widths = []
    # Ticks every 0.05, then every 2: the labels narrow from "-0.10" to "-2"
    for y_range in ((-0.12, 0.12), (-3.0, 3.0)):
        plot.getViewBox().setRange(yRange=y_range, padding=0)
        # The axis sizes itself to the tick labels when it paints them
        view.grab()
        qapp.processEvents()
        widths.append(axis.width())

    assert widths[1] == widths[0]


def test_general_tab_columns_are_equally_wide(qapp):
    window = autotune.Window()
    window.resize(1400, 1000)
    window.show()
    qapp.processEvents()

    widths = [
        plot.width()
        for plot in (window.pz_plot, window.step_plot)
        + (window.bode_mag_plot, window.bode_phase_plot)
    ]

    assert widths == pytest.approx([widths[0]] * 4, abs=1)
