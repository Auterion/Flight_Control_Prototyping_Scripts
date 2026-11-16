"""Tests of the model check: its verdict on simulated loops and on the sample
logs, and the widgets showing it.

Run with:  QT_QPA_PLATFORM=offscreen poetry run pytest test_model_check.py
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import presets  # noqa: E402
import pytest  # noqa: E402
from check_widgets import kNoModelCheckText, modelCheckHtml  # noqa: E402
from checks import (  # noqa: E402
    hertz,
    kBad,
    kCaution,
    kFail,
    kGood,
    kPass,
    kUnchecked,
)
from data_selection_window import DataSelectionWindow  # noqa: E402
from loop_model import LoopModel  # noqa: E402
from model_check import (  # noqa: E402
    ModelEvidence,
    checkModel,
    compareInBand,
    feedbackCorrelation,
    kFeedbackCorrelation,
    measurePlant,
    worstPhaseMargin,
)
from pid_design import computePidGmvc  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402
from scipy.signal import chirp  # noqa: E402
from system_identification import arx_transfer_function  # noqa: E402

import autotune  # noqa: E402

DT = 0.008
# ARX 2/2/1 fitted to the roll sweep of the x500 log
NUM = [0.15495, 0.18189, 0.01111]
DEN = [1.0, -1.8218, 0.82124]
DELAY = 1
MODEL = (NUM, DEN, DELAY)
FLOWN_GAINS = {"P": 0.07, "I": 2.4, "D": 0.002, "FF": 0.0}
SLOW_GAINS = {"P": 0.02, "I": 1.0, "D": 0.0, "FF": 0.0}
GOOD_FIT = 90.0
MAX_LAG = 25


def fast_gains():
    kc, ki, kd = computePidGmvc(NUM, DEN, DT, 0.13, 0.0, 0.5, DELAY)
    return {"P": kc, "I": ki, "D": kd, "FF": 0.0}


def closed_loop(plant, gains, r, noise_std=0.002, seed=0):
    """Input and output logged in the loop of LoopModel, which feeds the
    output back one sample later, with a white equation error."""
    num, den, delay = plant
    noise = noise_std * np.random.default_rng(seed).standard_normal(len(r))
    derivative_time = 1 / (2 * np.pi * 10.0)
    y = np.zeros(len(r))
    u = np.zeros(len(r))
    integral = derivative = previous_error = previous_feedback = 0.0
    for k in range(len(r)):
        y[k] = noise[k]
        y[k] -= sum(den[i] * y[k - i] for i in range(1, len(den)) if k >= i)
        y[k] += sum(b * u[k - delay - i] for i, b in enumerate(num) if k >= delay + i)
        feedback = y[k - 1] if k > 0 else 0.0
        error = r[k] - feedback
        integral += gains["I"] * DT / 2 * (error + previous_error)
        derivative = (
            (derivative_time - DT) * derivative
            - gains["D"] * (feedback - previous_feedback)
        ) / derivative_time
        u[k] = gains["P"] * (error + integral + derivative) + gains["FF"] * r[k]
        previous_error, previous_feedback = error, feedback
    return u, y


def sweep(f_start=0.5, f_stop=15.0, seconds=8.0):
    t = np.arange(0, seconds, DT)
    return 0.6 * chirp(t, f_start, t[-1], f_stop, method="logarithmic")


def with_resonance(plant, f=9.0, damping=0.15):
    num, den, delay = plant
    angle = 2 * np.pi * f * DT
    radius = np.exp(-damping * angle)
    resonance = np.array([1.0, -2 * radius * np.cos(angle), radius**2])
    return np.array(num) * resonance.sum(), np.convolve(den, resonance), delay


def check(model, gains, u, y, r, prediction_fit=GOOD_FIT, **evidence):
    evidence.setdefault("feedback_correlation", 0.0)
    num, den, delay = model
    loop = LoopModel(num, den, DT, delay, gains)
    return checkModel(
        loop,
        loop.stabilityMargins(),
        ModelEvidence(measurePlant(u, y, DT, r), len(u), r is not None, **evidence),
        prediction_fit,
    )


def states(model_check):
    return {check.name: check.state for check in model_check.checks}


@pytest.fixture(scope="module")
def swept_loop():
    r = sweep()
    return (*closed_loop(MODEL, FLOWN_GAINS, r), r)


def test_simulated_loop_is_the_one_of_the_loop_model():
    r = sweep()
    t = np.arange(len(r)) * DT

    _, y = closed_loop(MODEL, FLOWN_GAINS, r, noise_std=0.0)

    expected = LoopModel(*MODEL[:2], DT, DELAY, FLOWN_GAINS).simulate(
        t, r, np.zeros_like(r)
    )
    assert y[:-1] == pytest.approx(expected[1:], abs=1e-9)


def test_true_model_of_a_swept_loop_is_good(swept_loop):
    model_check = check(MODEL, FLOWN_GAINS, *swept_loop)

    assert model_check.level == kGood
    assert model_check.deciding is None
    assert set(states(model_check).values()) == {kPass}
    assert model_check.critical.is_crossover
    assert model_check.critical.f == pytest.approx(2.3, abs=0.1)
    assert model_check.band == pytest.approx(
        (model_check.critical.f / 3, 3 * model_check.critical.f)
    )


def test_model_with_twice_the_gain_is_not_to_tune_on(swept_loop):
    wrong_model = ([2 * b for b in NUM], DEN, DELAY)

    model_check = check(wrong_model, FLOWN_GAINS, *swept_loop)

    assert model_check.level == kBad
    assert model_check.deciding.name == "Model vs measured plant"
    loop = LoopModel(*wrong_model[:2], DT, DELAY, FLOWN_GAINS)
    comparison = compareInBand(
        loop, measurePlant(*swept_loop[:2], DT, swept_loop[2]), model_check.critical
    )
    assert comparison.gain_error == pytest.approx(6.0, abs=0.5)
    assert comparison.phase_error < 5.0


def test_slightly_wrong_model_is_a_caution(swept_loop):
    wrong_model = ([1.25 * b for b in NUM], DEN, DELAY)

    model_check = check(wrong_model, FLOWN_GAINS, *swept_loop)

    assert model_check.level == kCaution
    assert states(model_check)["Model vs measured plant"] == kCaution


def test_missed_delay_only_matters_for_a_fast_loop():
    plant = (NUM, DEN, DELAY + 1)
    r = sweep(0.2, 15.0, seconds=16.0)
    u, y = closed_loop(plant, SLOW_GAINS, r)

    slow = check(MODEL, SLOW_GAINS, u, y, r)
    fast = check(MODEL, dict(SLOW_GAINS, P=0.15), u, y, r)

    assert slow.critical.f < 1.0 < 3.0 < fast.critical.f
    assert slow.level == kGood
    assert fast.level == kBad
    assert fast.deciding.name == "Model vs measured plant"


def test_resonance_missed_by_the_model_fails_the_phase_margin():
    plant = with_resonance(MODEL)
    r = sweep()
    u, y = closed_loop(plant, FLOWN_GAINS, r)
    gains = fast_gains()
    assert LoopModel(*MODEL[:2], DT, DELAY, gains).isStable()
    assert not LoopModel(*plant[:2], DT, DELAY, gains).isStable()

    model_check = check(MODEL, gains, u, y, r)

    assert model_check.level == kBad
    assert states(model_check)["Margins on the measured plant"] == kFail
    assert "phase margin" in model_check.checks[3].reason


def test_crossover_above_the_excited_band_is_a_caution():
    r = sweep(0.5, 3.0)
    u, y = closed_loop(MODEL, FLOWN_GAINS, r)

    model_check = check(MODEL, fast_gains(), u, y, r)

    assert model_check.level == kCaution
    assert model_check.deciding.name == "Excitation"
    assert states(model_check)["Model vs measured plant"] == kUnchecked
    assert states(model_check)["Margins on the measured plant"] == kUnchecked


@pytest.mark.parametrize(
    "prediction_fit, state, level",
    [(75.0, kPass, kGood), (65.0, kCaution, kCaution), (50.0, kFail, kBad)],
)
def test_prediction_fit_is_a_floor(swept_loop, prediction_fit, state, level):
    model_check = check(MODEL, FLOWN_GAINS, *swept_loop, prediction_fit)

    assert states(model_check)["Prediction fit"] == state
    assert model_check.level == level


def test_prediction_fit_names_the_range_it_was_judged_on(swept_loop):
    estimation = check(MODEL, FLOWN_GAINS, *swept_loop)
    validation = check(MODEL, FLOWN_GAINS, *swept_loop, has_validation=True)

    assert "estimation range" in estimation.checks[1].value
    assert "validation range" in validation.checks[1].value


def test_range_too_short_to_measure_the_plant_is_not_checked(swept_loop):
    u, y, r = (signal[:200] for signal in swept_loop)

    model_check = check(MODEL, FLOWN_GAINS, u, y, r)

    assert measurePlant(u, y, DT, r) is None
    assert model_check.level == kCaution
    assert states(model_check)["Excitation"] == kUnchecked
    assert "200 samples" in model_check.checks[0].value


def test_range_too_short_to_resolve_the_band_is_a_caution():
    r = sweep(0.2, 15.0, seconds=4.0)
    u, y = closed_loop(MODEL, SLOW_GAINS, r)

    model_check = check(MODEL, SLOW_GAINS, u, y, r)

    assert model_check.critical.f < 1.0
    assert model_check.level == kCaution
    assert model_check.deciding.name == "Excitation"
    assert "too short" in model_check.deciding.reason


# First-order lag and delay of unit gain, which a feedback gain below 1 never
# makes cross 1: the integrator does, far below the lag, with over 90 deg of
# phase margin
LAG_PLANT = ([0.1], [1.0, -0.9], 8)
LOW_GAINS = {"P": 0.9, "I": 1.0, "D": 0.0, "FF": 0.0}


def test_loop_safe_at_its_crossover_is_checked_where_it_is_closest_to_minus_one():
    r = sweep()
    u, y = closed_loop(LAG_PLANT, LOW_GAINS, r)
    loop = LoopModel(*LAG_PLANT[:2], DT, LAG_PLANT[2], LOW_GAINS)
    _, phase_margin, modulus_margin, _, crossover, closest = loop.stabilityMargins()
    assert phase_margin > 90.0 and modulus_margin < 0.75

    model_check = check(LAG_PLANT, LOW_GAINS, u, y, r)
    wrong_model = ([0.2], *LAG_PLANT[1:])

    assert not model_check.critical.is_crossover
    assert model_check.critical.f == pytest.approx(closest / (2 * np.pi))
    assert model_check.critical.f > 5 * crossover / (2 * np.pi)
    assert model_check.level == kGood
    assert "phase margin" not in model_check.checks[3].value
    assert check(wrong_model, LOW_GAINS, u, y, r).level == kBad


def test_loop_far_from_minus_one_needs_no_excitation(swept_loop):
    open_loop = {"P": 1e-6, "I": 0.0, "D": 0.0, "FF": 0.0}
    wrong_model = ([2 * b for b in NUM], DEN, DELAY)

    model_check = check(MODEL, open_loop, *swept_loop)
    wrong = check(wrong_model, open_loop, *swept_loop)

    assert model_check.critical is None
    assert model_check.band is None
    assert model_check.level == kGood
    assert model_check.checks[0].value.startswith("Not needed with these gains")
    assert model_check.checks[2].value.startswith("Over the measured frequencies")
    assert wrong.level == kBad
    assert wrong.deciding.name == "Model vs measured plant"


def test_feedback_correlates_the_residuals_with_later_inputs(swept_loop):
    u, y, _ = swept_loop
    u_hover, y_hover = closed_loop(MODEL, FLOWN_GAINS, np.zeros(1000), noise_std=0.05)

    excited = feedbackCorrelation(*MODEL, u - u.mean(), y - y.mean(), MAX_LAG)
    hover = feedbackCorrelation(
        *MODEL, u_hover - u_hover.mean(), y_hover - y_hover.mean(), MAX_LAG
    )

    assert excited < 0.1 < kFeedbackCorrelation < hover


def test_closed_loop_data_without_a_reference_cannot_be_good(swept_loop):
    u, y, _ = swept_loop

    model_check = check(MODEL, FLOWN_GAINS, u, y, None, feedback_correlation=0.3)

    assert states(model_check)["Excitation"] == kUnchecked
    assert model_check.level != kGood


def test_excited_data_without_a_reference_can_be_good(swept_loop):
    u, y, _ = swept_loop

    model_check = check(MODEL, FLOWN_GAINS, u, y, None, feedback_correlation=0.05)

    assert model_check.level == kGood


def test_flown_gains_are_checked_when_known(swept_loop):
    unknown = check(MODEL, FLOWN_GAINS, *swept_loop)
    stable = check(MODEL, FLOWN_GAINS, *swept_loop, flown_loop_is_stable=True)
    unstable = check(MODEL, FLOWN_GAINS, *swept_loop, flown_loop_is_stable=False)

    assert "Flown gains on the model" not in states(unknown)
    assert states(stable)["Flown gains on the model"] == kPass
    assert stable.level == kGood
    assert unstable.level == kBad
    assert unstable.deciding.name == "Flown gains on the model"


def test_failed_check_decides_before_a_caution(swept_loop):
    wrong_model = ([2 * b for b in NUM], DEN, DELAY)

    model_check = check(wrong_model, FLOWN_GAINS, *swept_loop, prediction_fit=65.0)

    assert states(model_check)["Prediction fit"] == kCaution
    assert model_check.deciding.name == "Model vs measured plant"


def test_worst_phase_margin_is_the_margin_of_the_loop():
    loop = LoopModel(NUM, DEN, DT, DELAY, FLOWN_GAINS)
    f = np.geomspace(0.2, 40.0, 400)

    margin = worstPhaseMargin(f, loop.loop_gain(np.exp(2j * np.pi * f * DT)))

    assert margin == pytest.approx(loop.stabilityMargins()[1], abs=0.1)


def test_gain_that_never_crosses_one_has_no_phase_margin():
    f = np.geomspace(0.2, 40.0, 50)
    assert worstPhaseMargin(f, np.full(len(f), 0.5 + 0j)) is None


def test_details_list_each_check_and_link_the_plot_of_those_not_passed(swept_loop):
    wrong_model = ([2 * b for b in NUM], DEN, DELAY)
    model_check = check(wrong_model, FLOWN_GAINS, *swept_loop)

    details = modelCheckHtml(model_check)

    for name in states(model_check):
        assert name in details
    assert "✕ Do not tune" in details
    low, high = (hertz(f) for f in model_check.band)
    assert f"The checks look at the band {low}–{high} Hz" in details
    assert f"the {hertz(model_check.critical.f)} Hz crossover of the loop" in details
    assert details.count('<a href="bode">') == 1
    assert "<a href" not in modelCheckHtml(model_check, with_links=False)


def test_checks_say_what_they_found_and_what_to_do():
    r = sweep(0.5, 4.0)
    u, y = closed_loop(MODEL, FLOWN_GAINS, r)

    excitation = check(MODEL, FLOWN_GAINS, u, y, r).checks[0]

    assert excitation.state == kCaution
    assert excitation.value.startswith("The data excite only ")
    assert "% of the band that matters, 0.78–7.0 Hz" in excitation.value
    assert "coherence is at least 0.6" in excitation.criterion
    assert excitation.hint.startswith("Select a range with more excitation")
    assert excitation.reason.startswith("the data excite only ")


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
    window.combo_preset.setCurrentIndex(window.combo_preset.findText("Rollrate"))
    return window


# Roll of the x500 log: the PX4 autotune excitation, and pilot inputs
SWEPT_RANGE = (76.5, 83.5)
WEAKLY_EXCITED_RANGE = (41.0, 46.0)


def load_log(selection, monkeypatch, t_range):
    selection.t_start, selection.t_stop = t_range
    selection.exec_ = lambda: selection.loadSelection() or True
    monkeypatch.setattr(autotune, "DataSelectionWindow", lambda *args: selection)
    window = autotune.Window()
    window.loadLog()
    return window


def model_warning(window):
    """Warning of the model shown over the gains, None without one."""
    banner = window.checks_banner
    if banner.isHidden():
        return None
    for sentence in banner.label.text().split(" · "):
        if sentence.startswith(("⚠ Caution", "✕ Do not tune")):
            return sentence
    return None


def are_gains_struck(window):
    struck = {
        edit.font().strikeOut()
        for edits in window.gain_edit.values()
        for edit in edits.values()
    }
    assert len(struck) == 1
    return struck.pop()


def test_window_without_data_checks_no_model(qapp):
    window = autotune.Window()
    window.num, window.den, window.dt = NUM, DEN, DT
    window.sys_id_delays = DELAY
    window.Gz = arx_transfer_function(NUM, DEN, DT)
    window.is_system_identified = True
    window.gains = dict(FLOWN_GAINS)

    window.updateClosedLoop()

    assert window.model_check is None
    assert window.model_check_badge.text() == "—"
    assert model_warning(window) is None
    assert kNoModelCheckText in window.checks_panel.text.toPlainText()
    assert not are_gains_struck(window)


def test_model_of_the_swept_range_is_good(selection, monkeypatch):
    window = load_log(selection, monkeypatch, SWEPT_RANGE)

    assert window.model_check.level == kGood
    assert window.model_check_badge.text() == "✓ Good ▸"
    assert model_warning(window) is None
    assert not are_gains_struck(window)
    assert "Excitation" in window.model_check_badge.toolTip()
    assert "Flown gains on the model" in window.checks_panel.text.toPlainText()


def test_model_of_the_weakly_excited_range_is_not_to_tune_on(selection, monkeypatch):
    window = load_log(selection, monkeypatch, WEAKLY_EXCITED_RANGE)
    banner = window.checks_banner

    assert window.model_check.level == kBad
    assert window.model_check_badge.text() == "✕ Do not tune ▸"
    assert model_warning(window) == (
        f"✕ Do not tune: {window.model_check.deciding.reason}"
    )
    assert banner.label.text().startswith("✕ Do not tune: ")
    assert not banner.btn_override.isHidden()
    assert are_gains_struck(window)


def test_use_anyway_shows_the_gains_as_usable_until_the_next_model(
    selection, monkeypatch
):
    window = load_log(selection, monkeypatch, WEAKLY_EXCITED_RANGE)
    banner = window.checks_banner

    banner.btn_override.click()

    assert not are_gains_struck(window)
    assert banner.btn_override.isHidden()
    assert not banner.isHidden()

    window.gains["P"] *= 1.1
    window.updateClosedLoop()
    assert not are_gains_struck(window)

    window.onSysIdClicked()
    assert are_gains_struck(window)
    assert not banner.btn_override.isHidden()


def test_verdict_follows_the_gains(selection, monkeypatch):
    window = load_log(selection, monkeypatch, SWEPT_RANGE)
    flown_p = window.gains["P"]

    # A crossover far below the frequencies of the excitation
    window.gains["P"] = 0.01 * flown_p
    window.updateClosedLoop()

    assert window.model_check.level == kCaution
    assert window.model_check.deciding.name == "Excitation"
    assert model_warning(window).startswith("⚠ Caution: ")
    assert window.checks_banner.btn_override.isHidden()
    assert not are_gains_struck(window)

    window.gains["P"] = flown_p
    window.updateClosedLoop()

    assert window.model_check.level == kGood
    assert model_warning(window) is None


def test_prediction_horizon_updates_the_check(selection, monkeypatch):
    window = load_log(selection, monkeypatch, SWEPT_RANGE)
    assert window.model_check.checks[1].state == kPass

    # A horizon of seconds is the simulation, which drifts for a rate loop
    window.horizon_spinbox.setValue(5.0)

    assert window.model_check.checks[1].state == kFail
    assert window.lbl_fit.text() in window.model_check.checks[1].value


def test_selecting_other_data_forgets_the_check(selection, monkeypatch):
    window = load_log(selection, monkeypatch, WEAKLY_EXCITED_RANGE)

    window.reset()

    assert window.model_check is None
    assert window.model_check_badge.text() == "—"
    assert window.checks_banner.isHidden()
    assert not are_gains_struck(window)


def test_badge_and_banner_open_the_details(selection, monkeypatch):
    window = load_log(selection, monkeypatch, WEAKLY_EXCITED_RANGE)
    panel = window.checks_panel
    assert panel.isHidden()

    window.model_check_badge.click()
    assert not panel.isHidden()

    panel.hide()
    window.checks_banner.btn_details.click()
    assert not panel.isHidden()
    assert "Model vs measured plant" in panel.text.toPlainText()


def bode_band(window):
    """Bands shaded on the Bode magnitude and phase plots, in Hz."""
    return [
        tuple(10 ** np.array(band.getRegion())) if band.isVisible() else None
        for band in window.bode_check_bands
    ]


def bode_legend_names(window):
    return [label.text for _, label in window.bode_legend.items]


def test_bode_plot_shades_the_band_of_the_check(selection, monkeypatch):
    window = load_log(selection, monkeypatch, SWEPT_RANGE)
    band = window.model_check.band

    assert bode_band(window) == [pytest.approx(band)] * 2
    assert "Band of the model check" in bode_legend_names(window)

    window.gains["P"] *= 0.5
    window.updateClosedLoop()

    moved = window.model_check.band
    assert moved[1] < band[1]
    assert bode_band(window) == [pytest.approx(moved)] * 2

    window.reset()

    assert bode_band(window) == [None, None]
    assert "Band of the model check" not in bode_legend_names(window)


def test_details_link_zooms_the_bode_plot_on_the_band(selection, monkeypatch):
    window = load_log(selection, monkeypatch, WEAKLY_EXCITED_RANGE)
    window.plot_tabs.setCurrentWidget(window.robustness_plots)
    f_low, f_high = window.model_check.band

    window.checks_panel.evidenceRequested.emit("bode")

    assert window.plot_tabs.currentWidget() is window.general_plots
    x_min, x_max = window.bode_mag_plot.getViewBox().viewRange()[0]
    assert 10**x_min == pytest.approx(f_low / 2)
    assert 10**x_max == pytest.approx(2 * f_high)
    for plot, curves in (
        (window.bode_mag_plot, window.bode_mag_curves),
        (window.bode_phase_plot, window.bode_phase_curves),
    ):
        y_min, y_max = plot.getViewBox().viewRange()[1]
        f, measured = curves["measured"].xData, curves["measured"].yData
        in_view = measured[(f >= f_low / 2) & (f <= 2 * f_high)]
        assert len(in_view) > 0
        assert y_min < in_view.min() and in_view.max() < y_max


def test_details_link_shows_the_residuals(selection, monkeypatch):
    window = load_log(selection, monkeypatch, WEAKLY_EXCITED_RANGE)

    window.checks_panel.evidenceRequested.emit("residuals")

    assert window.plot_tabs.currentWidget() is window.residual_plots


def test_details_are_placed_once_then_stay_where_they_are_put(
    selection, monkeypatch, qapp
):
    window = load_log(selection, monkeypatch, SWEPT_RANGE)
    window.show()
    qapp.processEvents()
    panel = window.checks_panel

    window.showChecksPanel("model")
    panel.move(123, 45)
    panel.hide()
    window.showChecksPanel("model")

    assert (panel.x(), panel.y()) == (123, 45)


def test_banner_leaves_the_tabs_their_width(selection, monkeypatch, qapp):
    window = load_log(selection, monkeypatch, WEAKLY_EXCITED_RANGE)
    window.show()
    qapp.processEvents()
    tabs = window.tuning_tabs
    corner = window.checks_corner

    assert corner.sizeHint().width() <= tabs.width() - tabs.tabBar().sizeHint().width()
    assert corner.height() <= tabs.tabBar().height()


def test_fit_colors_follow_the_thresholds_of_the_check(qapp):
    window = autotune.Window()

    autotune.showFit(window.lbl_fit, 75.0)
    assert "green" in window.lbl_fit.styleSheet()
    autotune.showFit(window.lbl_fit, 65.0)
    assert "orange" in window.lbl_fit.styleSheet()
    autotune.showFit(window.lbl_fit, 55.0)
    assert "red" in window.lbl_fit.styleSheet()
