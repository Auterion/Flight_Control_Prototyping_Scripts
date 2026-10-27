"""Functional tests for the gain form selection of the PID tab.

Run with:  QT_QPA_PLATFORM=offscreen poetry run pytest test_pid_gains.py
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from loop_model import parallelGains  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

import autotune  # noqa: E402

IDEAL_GAINS = {"P": 0.2, "I": 2.5, "D": 0.004, "FF": 0.1}


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def window(qapp):
    # Window reuses pyplot figures 1 and 2, which still point to the Qt canvas
    # of any previous window
    plt.close("all")
    window = autotune.Window()
    window.show()
    window.activateWindow()
    window.gains = dict(IDEAL_GAINS)
    window.updateKIDSliders()
    return window


def type_into(qapp, edit, text):
    edit.setFocus()
    qapp.processEvents()
    edit.setText(text)


def move_slider(qapp, slider, value):
    slider.setFocus()
    qapp.processEvents()
    slider.setValue(value)


def shown(window, form):
    return {gain: float(e.text()) for gain, e in window.gain_edit[form].items()}


def test_ideal_form_is_editable_by_default(window):
    assert window.gain_form_radio["ideal"].isChecked()
    assert not any(e.isReadOnly() for e in window.gain_edit["ideal"].values())
    assert all(e.isReadOnly() for e in window.gain_edit["parallel"].values())


def test_both_columns_show_the_gains_in_their_form(window):
    assert shown(window, "ideal") == pytest.approx(IDEAL_GAINS, abs=1e-3)
    assert shown(window, "parallel") == pytest.approx(
        parallelGains(IDEAL_GAINS), abs=1e-4
    )


def test_selecting_parallel_makes_it_editable(window):
    window.gain_form_radio["parallel"].setChecked(True)

    assert window.gain_form == "parallel"
    assert not any(e.isReadOnly() for e in window.gain_edit["parallel"].values())
    assert all(e.isReadOnly() for e in window.gain_edit["ideal"].values())
    assert window.gain_slider["I"].value() == pytest.approx(0.5, abs=0.01)
    assert window.gain_slider["D"].value() == pytest.approx(0.0008, abs=1e-4)


def test_editing_a_parallel_gain_drives_the_ideal_ones(qapp, window):
    window.gain_form_radio["parallel"].setChecked(True)

    type_into(qapp, window.gain_edit["parallel"]["I"], "1.0")

    assert window.gains["I"] == pytest.approx(1.0 / IDEAL_GAINS["P"])
    assert float(window.gain_edit["ideal"]["I"].text()) == pytest.approx(5.0)
    assert window.gain_slider["I"].value() == pytest.approx(1.0)


def test_editing_parallel_kp_keeps_the_parallel_ki_and_kd(qapp, window):
    window.gain_form_radio["parallel"].setChecked(True)
    parallel_before = parallelGains(IDEAL_GAINS)

    type_into(qapp, window.gain_edit["parallel"]["P"], "0.4")

    assert shown(window, "parallel") == pytest.approx(
        {**parallel_before, "P": 0.4}, abs=1e-4
    )
    assert window.gains["I"] == pytest.approx(parallel_before["I"] / 0.4)


def test_editing_ideal_kp_drives_the_parallel_ki_and_kd(qapp, window):
    type_into(qapp, window.gain_edit["ideal"]["P"], "0.4")

    assert window.gains["I"] == pytest.approx(IDEAL_GAINS["I"])
    assert float(window.gain_edit["parallel"]["I"].text()) == pytest.approx(
        0.4 * IDEAL_GAINS["I"], abs=1e-4
    )


def test_read_only_column_ignores_edits(qapp, window):
    type_into(qapp, window.gain_edit["parallel"]["I"], "3.0")
    assert window.gains == pytest.approx(IDEAL_GAINS)


def test_slider_changes_the_active_form(qapp, window):
    window.gain_form_radio["parallel"].setChecked(True)

    move_slider(qapp, window.gain_slider["D"], 0.002)

    assert window.gains["D"] == pytest.approx(0.002 / IDEAL_GAINS["P"])
    assert float(window.gain_edit["parallel"]["D"].text()) == pytest.approx(0.002)


def test_parallel_kp_of_zero_is_ignored(qapp, window):
    window.gain_form_radio["parallel"].setChecked(True)

    type_into(qapp, window.gain_edit["parallel"]["P"], "0")

    assert window.gains == pytest.approx(IDEAL_GAINS)


def test_zero_loop_gain_is_drawn(window):
    # Typing "0.2" in the ideal Kp goes through "0". Called directly: an
    # exception raised in a Qt slot does not fail the test
    window.num, window.den = [0.0, 0.05, 0.03], [1.0, -1.6, 0.65]
    window.is_system_identified = True
    window.gains["P"] = 0.0

    window.updateClosedLoop()
