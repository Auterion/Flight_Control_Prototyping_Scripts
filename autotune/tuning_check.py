"""Tuning check: do the gains leave enough stability margin?

The margins are those of the loop closed on the model: they hold as far as
the model does, which the model check tells.
"""

from dataclasses import dataclass

import numpy as np
from checks import Check, Verdict, degrees, hertz, kCaution, kFail, kPass, kUnchecked


@dataclass
class MarginLimits:
    """Smallest margins of a good tuning."""

    gain_db: float
    phase_deg: float
    modulus: float
    delay_samples: float


class TuningCheck(Verdict):
    """Stability of the loop and each of its margins against its limit."""


kSlowDown = "Slow the loop down: lower Kp, then Ki."


def checkStability(is_stable):
    name = "Stability"
    if is_stable:
        return Check(
            name, kPass, "The loop closed on the model is stable with these gains."
        )
    return Check(
        name,
        kFail,
        "The loop closed on the model is unstable with these gains.",
        reason="the loop is unstable with these gains",
        hint="Lower the gains: Kp first, then Ki.",
        evidence="robustness",
    )


def checkMargin(name, value, criterion, is_enough, reason, hint=kSlowDown):
    if is_enough:
        return Check(name, kPass, value, criterion)
    return Check(
        name, kCaution, value, criterion, reason, hint=hint, evidence="robustness"
    )


def checkGainMargin(gain_margin, phase_crossover, limit_db):
    name = "Gain margin"
    criterion = f"Good from {limit_db:g} dB."
    if not np.isfinite(gain_margin) or gain_margin <= 0:
        return Check(
            name, kPass, "No increase of the loop gain makes the loop unstable."
        )
    gain_margin_db = 20 * np.log10(gain_margin)
    return checkMargin(
        name,
        f"The loop gain can be multiplied by {gain_margin:.1f} "
        f"({gain_margin_db:.1f} dB) before the loop is unstable, at "
        f"{hertz(phase_crossover / (2 * np.pi))} Hz.",
        criterion,
        gain_margin_db >= limit_db,
        f"gain margin of {gain_margin_db:.1f} dB (minimum {limit_db:g} dB)",
        hint="Lower Kp.",
    )


def checkPhaseMargin(phase_margin, gain_crossover, limit_deg):
    name = "Phase margin"
    criterion = f"Good from {limit_deg:g}°."
    if not np.isfinite(phase_margin):
        return Check(
            name,
            kPass,
            "The loop gain stays below 1 at all frequencies: no phase lag makes "
            "the loop unstable.",
        )
    return checkMargin(
        name,
        f"The loop tolerates {degrees(phase_margin)} of additional phase lag at "
        f"its {hertz(gain_crossover / (2 * np.pi))} Hz crossover before it is "
        "unstable.",
        criterion,
        phase_margin >= limit_deg,
        f"phase margin of {degrees(phase_margin)} (minimum {limit_deg:g}°)",
    )


def checkModulusMargin(modulus_margin, modulus_frequency, limit):
    peak_db = -20 * np.log10(modulus_margin)
    limit_db = -20 * np.log10(limit)
    return checkMargin(
        "Modulus margin",
        f"The loop gain stays at a distance of {modulus_margin:.2f} from −1, the "
        f"point of instability: it is the closest at "
        f"{hertz(modulus_frequency / (2 * np.pi))} Hz (a sensitivity peak of "
        f"{peak_db:.1f} dB).",
        f"Good from {limit:g}, a sensitivity peak of {limit_db:.0f} dB at most.",
        modulus_margin >= limit,
        f"modulus margin of {modulus_margin:.2f} (minimum {limit:g})",
    )


def checkDelayMargin(delay_margin, dt, limit_samples):
    name = "Delay margin"
    criterion = (
        f"Good from {limit_samples:g} sample of the model: its delay is a whole "
        "number of samples, and may be underestimated by one."
    )
    if not np.isfinite(delay_margin):
        return Check(name, kPass, "No additional delay makes the loop unstable.")
    samples = delay_margin / dt
    return checkMargin(
        name,
        f"The loop tolerates {delay_margin * 1e3:.0f} ms of additional delay, "
        f"{samples:.1f} samples of the model, before it is unstable.",
        criterion,
        samples >= limit_samples,
        f"delay margin of {samples:.1f} samples (minimum {limit_samples:g})",
        hint="Slow the loop down: a lower crossover tolerates more delay.",
    )


def checkTuning(loop, stability_margins, is_stable, limits):
    """TuningCheck of the gains of loop, a LoopModel whose stabilityMargins()
    are stability_margins, against the MarginLimits."""
    if not is_stable:
        margins = [
            Check(name, kUnchecked, "Does not apply to an unstable loop.")
            for name in (
                "Gain margin",
                "Phase margin",
                "Modulus margin",
                "Delay margin",
            )
        ]
        return TuningCheck([checkStability(False), *margins])

    (
        gain_margin,
        phase_margin,
        modulus_margin,
        phase_crossover,
        gain_crossover,
        modulus_frequency,
    ) = stability_margins
    delay_margin, _ = loop.delayMargin()
    return TuningCheck(
        [
            checkStability(True),
            checkGainMargin(gain_margin, phase_crossover, limits.gain_db),
            checkPhaseMargin(phase_margin, gain_crossover, limits.phase_deg),
            checkModulusMargin(modulus_margin, modulus_frequency, limits.modulus),
            checkDelayMargin(delay_margin, loop.dt, limits.delay_samples),
        ]
    )
