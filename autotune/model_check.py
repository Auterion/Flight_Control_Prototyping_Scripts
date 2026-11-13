"""Model check: is the identified model good enough to tune these gains on?

A controller designed on a wrong model can be unstable on the vehicle. The
model is compared with the plant measured from the data (its frequency
response, estimated without a model structure) where the stability of the
loop closed with the gains is at stake: around its critical frequency. The
verdict therefore follows the gains.
"""

from dataclasses import dataclass

import numpy as np
from system_identification import (
    kMinCoherenceSamples,
    kMinGoodCoherence,
    measured_frequency_response,
    residual_correlations,
)

kPass = "pass"
kCaution = "caution"
kFail = "fail"
# Could not be checked: the model is not known to be good
kUnchecked = "unchecked"

kGood = "good"
kDoNotTune = "do_not_tune"

# With more phase margin, the loop gain crosses 1 with a positive real part:
# farther from -1 than without feedback. The stability of such a loop, e.g. a
# low feedback gain next to a feedforward, is not at stake at its crossover
kMaxCrossoverPhaseMargin = 90.0
# Modulus margin from which such a loop has no critical frequency: at -180
# deg of phase, a plant of 4 times the gain of the model would not reach -1
kSafeModulusMargin = 0.75
# The stability margins are set by the plant from about 0.3 to 3 times the
# crossover, where a model for control design has to be accurate (Tischler's
# rule of thumb)
kBandFactor = 3.0
kMinBandPoints = 4
kMinBandCoverage = 0.7

# The thresholds below separate the well excited ranges of the sample logs
# (sweeps, PX4 autotune excitation) from the weakly excited ones and from the
# biased models of vtol_standard.ulg, and the models of simulated loops whose
# phase margin is right within 10 deg from the others.
kGoodFit = 70.0
kPoorFit = 60.0
kGoodGainError = 1.0  # dB
kGoodPhaseError = 8.0  # deg
kPoorGainError = 3.0
kPoorPhaseError = 15.0
# Phase margin of the model above the measured one, deg
kGoodMarginOptimism = 10.0
kPoorMarginOptimism = 20.0
# Share of the modulus margin of the model missing on the measured plant
kGoodModulusLoss = 0.3
kPoorModulusLoss = 0.5
# Correlation of the residuals with later inputs above which the input mostly
# reacts to the output: 0.05 to 0.08 on the well excited ranges, 0.18 to 0.26
# on the weakly excited ones
kFeedbackCorrelation = 0.15


@dataclass
class MeasuredPlant:
    """Frequency response of the plant measured from the data."""

    f: np.ndarray  # Hz
    response: np.ndarray
    is_coherent: np.ndarray


@dataclass
class ModelEvidence:
    """What the data say about a model, whatever the gains.

    measured is None when the range is too short to measure the plant,
    flown_loop_is_stable when the flown gains are not known.
    """

    measured: MeasuredPlant | None
    num_samples: int
    has_reference: bool
    feedback_correlation: float
    flown_loop_is_stable: bool | None = None
    has_validation: bool = False


@dataclass
class Check:
    """One check of the model: value is what it found, criterion what it
    takes to pass, reason the finding in a few words, hint what to change
    when it does not pass and evidence the plot showing it."""

    name: str
    state: str
    value: str
    criterion: str = ""
    reason: str = ""
    hint: str = ""
    evidence: str | None = None


@dataclass
class CriticalFrequency:
    """Where the stability of a loop is at stake: its gain crossover or, for
    a loop safe there, where its gain comes the closest to -1."""

    f: float  # Hz
    is_crossover: bool

    @property
    def name(self):
        return "the crossover" if self.is_crossover else "the closest approach to −1"


def around(critical):
    """Where the model is compared with the measured plant, in words."""
    if critical is None:
        return "over the measured frequencies"
    return f"around {critical.name}"


@dataclass
class ModelCheck:
    """critical is the CriticalFrequency the model was checked around, band
    the (lowest, highest) frequencies in Hz it was checked over: both None
    for a loop whose gain stays far from -1 at all frequencies."""

    critical: CriticalFrequency | None
    band: tuple | None
    checks: list

    @property
    def level(self):
        states = {check.state for check in self.checks}
        if kFail in states:
            return kDoNotTune
        if states & {kCaution, kUnchecked}:
            return kCaution
        return kGood

    @property
    def deciding(self):
        """The check that sets the level, None when the model is good."""
        for state in (kFail, kCaution, kUnchecked):
            for check in self.checks:
                if check.state == state:
                    return check
        return None


@dataclass
class BandComparison:
    """Model against the measured plant over the coherent points of a band:
    num_points of the num_frequencies measured there. The phase margins are
    None without a crossover among these points."""

    f_low: float
    f_high: float
    num_frequencies: int
    num_points: int
    coverage: float
    is_bracketed: bool
    gain_error: float = np.nan
    phase_error: float = np.nan
    measured_margin: float | None = None
    model_margin: float | None = None
    measured_modulus: float = np.nan
    model_modulus: float = np.nan


def measurePlant(u, y, dt, reference=None):
    """MeasuredPlant of the data, None when they are too short."""
    if len(u) < kMinCoherenceSamples:
        return None
    f, response, gamma2 = measured_frequency_response(u, y, dt, reference)
    return MeasuredPlant(f, response, gamma2 >= kMinGoodCoherence)


def feedbackCorrelation(num, den, delay, u, y, max_lag):
    """Largest correlation of the residuals of the model with the current and
    later inputs, on u and y preprocessed as for the fit: the input reacting
    to the output, as a controller does in closed loop."""
    _, cross, _ = residual_correlations(num, den, delay, u, y, max_lag)
    return float(np.max(np.abs(cross[: max_lag + 1])))


def worstPhaseMargin(f, loop_gain):
    """Smallest phase margin (deg) over the unit-gain crossings of the points,
    None when they do not cross it downwards.

    A resonance the model misses makes the measured loop cross again, with
    less margin than at the crossover of the model.
    """
    log_gain = np.log(np.abs(loop_gain))
    margins = []
    for i in np.nonzero((log_gain[:-1] >= 0) & (log_gain[1:] < 0))[0]:
        ratio = log_gain[i] / (log_gain[i] - log_gain[i + 1])
        phase = np.angle(loop_gain[i]) + ratio * np.angle(
            loop_gain[i + 1] / loop_gain[i]
        )
        margins.append(np.rad2deg(np.mod(phase + 2 * np.pi, 2 * np.pi) - np.pi))
    return float(min(margins)) if margins else None


def criticalFrequency(stability_margins):
    """CriticalFrequency of a loop from its LoopModel.stabilityMargins(), None
    when its gain stays far from -1."""
    _, phase_margin, modulus_margin, _, gain_crossover, modulus_frequency = (
        stability_margins
    )
    if np.isfinite(gain_crossover) and phase_margin < kMaxCrossoverPhaseMargin:
        return CriticalFrequency(gain_crossover / (2 * np.pi), is_crossover=True)
    if modulus_margin < kSafeModulusMargin and np.isfinite(modulus_frequency):
        return CriticalFrequency(modulus_frequency / (2 * np.pi), is_crossover=False)
    return None


def checkedBand(critical, dt):
    """(lowest, highest) frequencies in Hz the model is checked over, None
    when it is checked over all the measured frequencies."""
    if critical is None:
        return None
    return critical.f / kBandFactor, min(critical.f * kBandFactor, 0.5 / dt)


def compareInBand(loop, measured, critical):
    """BandComparison over the checkedBand of the loop."""
    band = checkedBand(critical, loop.dt)
    if band is None:
        in_band = np.ones(len(measured.f), dtype=bool)
        f_low, f_high = measured.f[0], 0.5 / loop.dt
    else:
        f_low, f_high = band
        in_band = (measured.f >= f_low) & (measured.f <= f_high)
    used = in_band & measured.is_coherent
    f = measured.f[used]
    comparison = BandComparison(
        f_low=f_low,
        f_high=f_high,
        num_frequencies=int(np.count_nonzero(in_band)),
        num_points=int(np.count_nonzero(used)),
        coverage=float(np.mean(measured.is_coherent[in_band])) if in_band.any() else 0,
        is_bracketed=critical is not None
        and bool(np.any(f <= critical.f) and np.any(f >= critical.f)),
    )
    if comparison.num_points == 0:
        return comparison

    z = np.exp(2j * np.pi * f * loop.dt)
    # The plant of the loop has one more sample of delay: the measurement's
    measured_to_model = measured.response[used] / (loop.plant(z) * z)
    comparison.gain_error = float(
        np.median(np.abs(20 * np.log10(np.abs(measured_to_model))))
    )
    comparison.phase_error = float(
        np.median(np.abs(np.rad2deg(np.angle(measured_to_model))))
    )
    model_loop = loop.loop_gain(z)
    measured_loop = model_loop * measured_to_model
    comparison.model_margin = worstPhaseMargin(f, model_loop)
    comparison.measured_margin = worstPhaseMargin(f, measured_loop)
    comparison.model_modulus = float(np.min(np.abs(1 + model_loop)))
    comparison.measured_modulus = float(np.min(np.abs(1 + measured_loop)))
    return comparison


def hertz(f):
    if f >= 10:
        return f"{f:.0f}"
    return f"{f:.1f}" if f >= 1 else f"{f:.2f}"


def degrees(angle):
    return f"{angle:.0f}°".replace("-", "−")


def bandText(comparison):
    return f"{hertz(comparison.f_low)}–{hertz(comparison.f_high)} Hz"


kNotMeasured = (
    "Cannot be checked: the plant is not measured reliably where the model "
    "has to be right (see Excitation)."
)


def checkExcitation(evidence, critical, comparison):
    name = "Excitation"
    if evidence.measured is None:
        return Check(
            name,
            kUnchecked,
            "The range is too short to measure the frequency response of the "
            f"plant: it has {evidence.num_samples} samples, at least "
            f"{kMinCoherenceSamples} are needed.",
            reason="the range is too short to measure the plant",
            hint="Select a longer range.",
        )
    if (
        not evidence.has_reference
        and evidence.feedback_correlation > kFeedbackCorrelation
    ):
        return Check(
            name,
            kUnchecked,
            "The excitation cannot be checked: the preset has no reference, and "
            "in these data the input mostly reacts to the output (closed loop). "
            "The plant measured from the input is then biased by the feedback.",
            reason="closed-loop data cannot be checked without a reference",
            hint="Add the reference signal of the loop to the preset.",
            evidence="residuals",
        )
    if critical is None:
        return Check(
            name,
            kPass,
            "Not needed with these gains: the loop gain stays far from −1 at "
            "all frequencies, so there is no band where the stability margins "
            "depend on the model.",
        )

    band = bandText(comparison)
    criterion = (
        f"Good from {kMinBandCoverage:.0%}. A frequency is measured reliably "
        f"when its coherence is at least {kMinGoodCoherence:g}: the black "
        "points of the Bode plot."
    )
    if comparison.num_frequencies < kMinBandPoints:
        return Check(
            name,
            kCaution,
            f"The range is too short to check the band that matters, {band}: "
            f"its frequency resolution gives {comparison.num_frequencies} of the "
            f"{kMinBandPoints} frequencies needed there.",
            reason=f"the range is too short to check the band {around(critical)}",
            hint="Select a longer range: its frequency resolution is finer.",
        )
    hint = (
        "Select a range with more excitation in this band (a sweep, the PX4 "
        "autotune excitation), or change the gains to move the band into the "
        "excited frequencies."
    )
    if comparison.num_points < kMinBandPoints or not comparison.is_bracketed:
        return Check(
            name,
            kCaution,
            f"The data do not excite the band that matters, {band}: the plant "
            f"is not measured reliably on both sides of {critical.name}, so "
            "the model cannot be verified there.",
            criterion,
            reason=f"the data do not excite the band {around(critical)} ({band})",
            hint=hint,
            evidence="bode",
        )
    if comparison.coverage < kMinBandCoverage:
        return Check(
            name,
            kCaution,
            f"The data excite only {comparison.coverage:.0%} of the band that "
            f"matters, {band}: at the other frequencies the plant is not "
            "measured reliably, so the model cannot be verified there.",
            criterion,
            reason=(
                f"the data excite only {comparison.coverage:.0%} of the band "
                f"{around(critical)} ({band})"
            ),
            hint=hint,
            evidence="bode",
        )
    return Check(
        name,
        kPass,
        f"The data excite the loop where it matters: {comparison.coverage:.0%} "
        f"of the frequencies of the band {band} are measured reliably.",
        criterion,
    )


def checkPredictionFit(prediction_fit, has_validation):
    name = "Prediction fit"
    data = "validation" if has_validation else "estimation"
    criterion = f"Good from {kGoodFit:.0f}%, fail below {kPoorFit:.0f}%."
    hint = (
        "Try other model orders or another delay (Find parameters). If the fit "
        "stays low, the range is probably not excited enough."
    )
    if not np.isfinite(prediction_fit):
        return Check(
            name,
            kFail,
            f"The model cannot be compared with the output of the {data} range.",
            criterion,
            reason="the model does not predict the output",
            hint=hint,
        )
    value = (
        f"The model predicts the logged output with a fit of {prediction_fit:.1f}% "
        f"on the {data} range (100% is a perfect prediction)."
    )
    if prediction_fit >= kGoodFit:
        return Check(name, kPass, value, criterion)
    return Check(
        name,
        kFail if prediction_fit < kPoorFit else kCaution,
        value,
        criterion,
        reason=f"the model predicts the output with a fit of {prediction_fit:.0f}% only",
        hint=hint,
    )


def isCompared(comparison):
    return comparison is not None and comparison.num_points >= kMinBandPoints


def checkPlant(critical, comparison):
    name = "Model vs measured plant"
    criterion = (
        f"Good up to {kGoodGainError:g} dB and {kGoodPhaseError:g}°, fail above "
        f"{kPoorGainError:g} dB or {kPoorPhaseError:g}° (median over the "
        "reliably measured frequencies)."
    )
    if not isCompared(comparison):
        return Check(
            name,
            kUnchecked,
            kNotMeasured,
            reason="the model could not be compared with the measured plant",
        )
    gain_error, phase_error = comparison.gain_error, comparison.phase_error
    where = around(critical)
    if critical is not None:
        where += f" ({bandText(comparison)})"
    value = (
        f"{where[0].upper()}{where[1:]}, the model differs from the plant "
        f"measured from the data by {gain_error:.1f} dB in gain and "
        f"{phase_error:.0f}° in phase."
    )
    if gain_error <= kGoodGainError and phase_error <= kGoodPhaseError:
        return Check(name, kPass, value, criterion)
    is_poor = gain_error > kPoorGainError or phase_error > kPoorPhaseError
    trust = "cannot be trusted" if is_poor else "are approximate"
    return Check(
        name,
        kFail if is_poor else kCaution,
        f"{value} The margins computed on this model {trust}.",
        criterion,
        reason=(
            f"the model is {gain_error:.1f} dB and {phase_error:.0f}° off the "
            f"plant measured {around(critical)}"
        ),
        hint=(
            "Try another delay or other model orders (Find parameters). If the "
            "gap stays, select a better excited range."
        ),
        evidence="bode",
    )


def checkMargins(critical, comparison):
    """The margins of the gains on the measured plant against those on the
    model, at the same points: the modulus margin, the distance of the loop
    gain to -1, and the phase margin when the loop gain crosses 1 there."""
    name = "Margins on the measured plant"
    criterion = (
        f"Good when the measured plant gives at most {kGoodMarginOptimism:g}° "
        f"less phase margin and {kGoodModulusLoss:.0%} less modulus margin "
        f"than the model, fail above {kPoorMarginOptimism:g}° or "
        f"{kPoorModulusLoss:.0%}."
    )
    if not isCompared(comparison):
        return Check(
            name,
            kUnchecked,
            kNotMeasured,
            reason="the margins could not be measured from the data",
        )

    modulus_loss = 1 - comparison.measured_modulus / comparison.model_modulus
    margins = (
        "a modulus margin (distance of the loop gain to −1) of "
        f"{comparison.measured_modulus:.2f}, the model {comparison.model_modulus:.2f}"
    )
    reason = (
        f"the modulus margin is {modulus_loss:.0%} lower on the measured plant "
        "than on the model"
    )
    # Shares of the poor thresholds, to tell the worse of both margins
    severity = modulus_loss / kPoorModulusLoss
    is_good = modulus_loss <= kGoodModulusLoss
    if comparison.measured_margin is not None and comparison.model_margin is not None:
        optimism = comparison.model_margin - comparison.measured_margin
        margins = (
            f"a phase margin of {degrees(comparison.measured_margin)}, the "
            f"model {degrees(comparison.model_margin)}; and {margins}"
        )
        is_good = is_good and optimism <= kGoodMarginOptimism
        if optimism / kPoorMarginOptimism > severity:
            severity = optimism / kPoorMarginOptimism
            reason = (
                f"the phase margin is {degrees(optimism)} lower on the measured "
                "plant than on the model"
            )
    value = f"With these gains, the plant measured from the data gives {margins}."
    if is_good:
        return Check(name, kPass, value, criterion)
    return Check(
        name,
        kFail if severity > 1 else kCaution,
        f"{value} The model is optimistic: the real loop is closer to "
        "instability than the plots show.",
        criterion,
        reason=reason,
        hint=f"Use slower gains, or improve the model {around(critical)}.",
        evidence="bode",
    )


def checkFlownLoop(is_stable):
    name = "Flown gains on the model"
    if is_stable:
        return Check(
            name,
            kPass,
            "The model is stable with the gains flown in the log, as the "
            "vehicle was.",
        )
    return Check(
        name,
        kFail,
        "The model is unstable with the gains flown in the log, although the "
        "vehicle flew with them.",
        reason="the model is unstable with the gains flown in the log",
        hint=(
            "The model, its delay or the PID options of the preset are wrong: "
            "check them."
        ),
    )


def checkModel(loop, stability_margins, evidence, prediction_fit):
    """ModelCheck of the model in loop, a LoopModel closed with the gains to
    check it for, stability_margins being its stabilityMargins()."""
    critical = criticalFrequency(stability_margins)
    comparison = None
    if evidence.measured is not None:
        comparison = compareInBand(loop, evidence.measured, critical)
    checks = [
        checkExcitation(evidence, critical, comparison),
        checkPredictionFit(prediction_fit, evidence.has_validation),
        checkPlant(critical, comparison),
        checkMargins(critical, comparison),
    ]
    if evidence.flown_loop_is_stable is not None:
        checks.append(checkFlownLoop(evidence.flown_loop_is_stable))
    return ModelCheck(critical, checkedBand(critical, loop.dt), checks)
