"""Tests of the tuning check: its verdict on the margins of a loop, and the
widgets showing it.

Run with:  QT_QPA_PLATFORM=offscreen poetry run pytest test_tuning_check.py
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from check_widgets import (  # noqa: E402
    kNoModelCheckText,
    kNoTuningCheckText,
    kTuning,
    tuningCheckHtml,
)
from checks import kBad, kCaution, kFail, kGood, kPass, kUnchecked  # noqa: E402
from loop_model import LoopModel  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402
from system_identification import arx_transfer_function  # noqa: E402
from tuning_check import MarginLimits, checkTuning  # noqa: E402

import autotune  # noqa: E402

DT = 0.008
# ARX 2/2/1 fitted to the roll sweep of the x500 log
NUM = [0.15495, 0.18189, 0.01111]
DEN = [1.0, -1.8218, 0.82124]
DELAY = 1
SLOW_GAINS = {"P": 0.05, "I": 1.0, "D": 0.004, "FF": 0.0}
# The gains flown in the x500 log
TIGHT_GAINS = {"P": 0.1431, "I": 2.362, "D": 0.0019, "FF": 0.0}
UNSTABLE_GAINS = {"P": 0.6, "I": 1.0, "D": 0.004, "FF": 0.0}
LIMITS = MarginLimits(gain_db=6.0, phase_deg=45.0, modulus=0.5, delay_samples=1.0)
MARGINS = ("Gain margin", "Phase margin", "Modulus margin", "Delay margin")


def check(gains, limits=LIMITS):
    loop = LoopModel(NUM, DEN, DT, DELAY, gains)
    return checkTuning(loop, loop.stabilityMargins(), loop.isStable(), limits)


def states(tuning_check):
    return {check.name: check.state for check in tuning_check.checks}


def test_tuning_with_enough_margins_is_good():
    tuning_check = check(SLOW_GAINS)

    assert tuning_check.level == kGood
    assert tuning_check.deciding is None
    assert states(tuning_check) == dict.fromkeys(("Stability", *MARGINS), kPass)


def test_tuning_with_too_little_margin_is_a_caution():
    loop = LoopModel(NUM, DEN, DT, DELAY, TIGHT_GAINS)
    _, phase_margin, modulus_margin, *_ = loop.stabilityMargins()
    assert phase_margin < 45.0 and modulus_margin < 0.5

    tuning_check = check(TIGHT_GAINS)

    assert tuning_check.level == kCaution
    assert states(tuning_check) == {
        "Stability": kPass,
        "Gain margin": kPass,
        "Phase margin": kCaution,
        "Modulus margin": kCaution,
        "Delay margin": kPass,
    }
    assert tuning_check.deciding.name == "Phase margin"
    assert tuning_check.deciding.reason == (
        f"phase margin of {phase_margin:.0f}° (minimum 45°)"
    )
    assert tuning_check.deciding.hint
    assert tuning_check.deciding.evidence == "robustness"


@pytest.mark.parametrize(
    "limits, margin",
    [
        (MarginLimits(40.0, 0.0, 0.01, 0.0), "Gain margin"),
        (MarginLimits(0.0, 80.0, 0.01, 0.0), "Phase margin"),
        (MarginLimits(0.0, 0.0, 0.95, 0.0), "Modulus margin"),
        (MarginLimits(0.0, 0.0, 0.01, 50.0), "Delay margin"),
    ],
)
def test_each_margin_is_checked_against_its_limit(limits, margin):
    tuning_check = check(SLOW_GAINS, limits)

    assert tuning_check.level == kCaution
    assert [name for name, state in states(tuning_check).items() if state != kPass] == [
        margin
    ]


def test_checks_give_the_margins_of_the_loop():
    loop = LoopModel(NUM, DEN, DT, DELAY, SLOW_GAINS)
    gain_margin, phase_margin, modulus_margin, *_ = loop.stabilityMargins()
    delay_margin, _ = loop.delayMargin()

    values = {check.name: check.value for check in check(SLOW_GAINS).checks}

    assert f"({20 * np.log10(gain_margin):.1f} dB)" in values["Gain margin"]
    assert f"tolerates {phase_margin:.0f}° of additional" in values["Phase margin"]
    assert f"distance of {modulus_margin:.2f} from −1" in values["Modulus margin"]
    assert f"{delay_margin / DT:.1f} samples" in values["Delay margin"]


def test_unstable_tuning_is_bad_and_its_margins_do_not_apply():
    assert not LoopModel(NUM, DEN, DT, DELAY, UNSTABLE_GAINS).isStable()

    tuning_check = check(UNSTABLE_GAINS)

    assert tuning_check.level == kBad
    assert tuning_check.deciding.name == "Stability"
    assert tuning_check.deciding.reason == "the loop is unstable with these gains"
    assert states(tuning_check) == {
        "Stability": kFail,
        **dict.fromkeys(MARGINS, kUnchecked),
    }


def test_loop_gain_below_one_has_margins_to_spare():
    # A stable plant: its loop gain stays below 1 with a low gain
    loop = LoopModel([0.1], [1.0, -0.9], DT, DELAY, dict(SLOW_GAINS, P=0.1, I=0.0))

    tuning_check = checkTuning(loop, loop.stabilityMargins(), loop.isStable(), LIMITS)

    assert tuning_check.level == kGood
    assert "stays below 1" in tuning_check.checks[2].value
    assert "No additional delay" in tuning_check.checks[4].value


def test_details_explain_the_verdict_and_link_the_margins_not_met():
    details = tuningCheckHtml(check(TIGHT_GAINS))

    assert "⚠ Low margins" in details
    assert "the loop closed on the model" in details
    assert details.count('<a href="robustness">') == 2
    assert "<a href" not in tuningCheckHtml(check(TIGHT_GAINS), with_links=False)
    assert "✕ Unstable" in tuningCheckHtml(check(UNSTABLE_GAINS))
    assert kNoTuningCheckText in tuningCheckHtml(None)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def window(qapp):
    window = autotune.Window()
    window.num, window.den, window.dt = NUM, DEN, DT
    window.sys_id_delays = DELAY
    window.Gz = arx_transfer_function(NUM, DEN, DT)
    window.is_system_identified = True
    set_gains(window, SLOW_GAINS)
    return window


def set_gains(window, gains):
    window.gains = dict(gains)
    window.updateClosedLoop()


def banner_text(window):
    banner = window.checks_banner
    return None if banner.isHidden() else banner.label.text()


def test_tuning_is_checked_without_data(window):
    assert window.model_check is None
    assert window.tuning_check.level == kGood
    assert window.tuning_check_badge.text() == "Tuning check: ✓ Good ▸"
    assert banner_text(window) is None
    details = window.checks_panel.text.toPlainText()
    assert kNoModelCheckText in details
    assert "the current gains keep the recommended stability margins" in details


def test_tuning_verdict_follows_the_gains(window):
    set_gains(window, TIGHT_GAINS)

    assert window.tuning_check_badge.text() == "Tuning check: ⚠ Low margins ▸"
    assert banner_text(window) == (
        f"⚠ Low margins: {window.tuning_check.deciding.reason}"
    )
    assert "Phase margin" in window.tuning_check_badge.toolTip()

    set_gains(window, UNSTABLE_GAINS)

    assert window.tuning_check_badge.text() == "Tuning check: ✕ Unstable ▸"
    assert banner_text(window) == "✕ Unstable: the loop is unstable with these gains"

    set_gains(window, SLOW_GAINS)

    assert window.tuning_check_badge.text() == "Tuning check: ✓ Good ▸"
    assert banner_text(window) is None


def test_tuning_check_uses_the_limits_of_the_plots(window):
    window.kMinPhaseMarginDeg = 80.0

    set_gains(window, SLOW_GAINS)

    assert window.tuning_check.level == kCaution
    assert window.tuning_check.deciding.name == "Phase margin"
    assert "(minimum 80°)" in banner_text(window)


def test_tuning_warning_does_not_strike_the_gains(window):
    set_gains(window, UNSTABLE_GAINS)

    banner = window.checks_banner
    assert banner.btn_override.isHidden()
    assert not any(
        edit.font().strikeOut()
        for edits in window.gain_edit.values()
        for edit in edits.values()
    )


def test_tuning_badge_opens_the_details_at_the_tuning_check(window, qapp):
    window.show()
    qapp.processEvents()
    shown = []
    window.checks_panel.showSection = shown.append

    window.tuning_check_badge.click()

    assert not window.checks_panel.isHidden()
    assert shown == [kTuning]


def test_banner_details_open_at_the_first_warning(window, qapp):
    window.show()
    qapp.processEvents()
    shown = []
    window.checks_panel.showSection = shown.append
    set_gains(window, TIGHT_GAINS)

    window.checks_banner.btn_details.click()

    assert shown == [kTuning]


def test_details_scroll_to_the_tuning_check(window, qapp):
    window.show()
    qapp.processEvents()
    panel = window.checks_panel
    panel.resize(panel.width(), 120)

    window.showChecksPanel(kTuning)
    qapp.processEvents()

    assert panel.text.verticalScrollBar().value() > 0


def test_margin_link_shows_the_robustness_plots(window):
    set_gains(window, TIGHT_GAINS)
    assert window.plot_tabs.currentWidget() is window.general_plots

    window.checks_panel.evidenceRequested.emit("robustness")

    assert window.plot_tabs.currentWidget() is window.robustness_plots


def test_new_model_forgets_the_tuning_check(window):
    window.reset()

    assert window.tuning_check is None
    assert window.tuning_check_badge.text() == "Tuning check: —"
    assert banner_text(window) is None
    assert kNoTuningCheckText in window.checks_panel.text.toPlainText()


class Verdict:
    """Stand-in for a check of a given level."""

    def __init__(self, level, reason):
        self.level = level
        self.deciding = type("Deciding", (), {"reason": reason})()


def test_banner_puts_the_worst_warning_first(window):
    banner = window.checks_banner

    banner.showChecks(Verdict(kCaution, "model"), Verdict(kBad, "tuning"), False)
    assert banner.label.text() == "✕ Unstable: tuning · ⚠ Caution: model"
    assert banner.warned_kind == kTuning
    assert banner.btn_override.isHidden()

    banner.showChecks(Verdict(kBad, "model"), Verdict(kCaution, "tuning"), False)
    assert banner.label.text() == "✕ Bad: model · ⚠ Low margins: tuning"
    assert not banner.btn_override.isHidden()

    banner.showChecks(Verdict(kCaution, "model"), Verdict(kCaution, "tuning"), False)
    assert banner.label.text() == "⚠ Caution: model · ⚠ Low margins: tuning"
