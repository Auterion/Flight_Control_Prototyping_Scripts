"""Unit tests for the closed-loop model (loop_model.py).

Run with:  poetry run pytest test_loop_model.py
"""

import control as ctrl
import numpy as np
import pytest
from loop_model import LoopModel, kDerivativeCutoffFreq
from system_identification import arx_transfer_function

DT = 0.005
NUM = [0.0, 0.05, 0.03]
DEN = [1.0, -1.6, 0.65]
GAINS = {"P": 0.8, "I": 0.3, "D": 0.02, "FF": 0.1}
UNSTABLE_GAINS = {"P": 1.5, "I": 0.5, "D": 0.08, "FF": 0.0}

OPTIONS = [
    pytest.param(dict(), id="default"),
    pytest.param(dict(p_on_feedback=True), id="p_on_feedback"),
    pytest.param(dict(d_on_feedback=False), id="d_on_error"),
    pytest.param(dict(negate_output=True), id="negated"),
]


def expected_loop_gain(delays, gains, negate_output=False):
    kc, ki, kd = gains["P"], gains["I"], gains["D"]
    tau = 1 / (2 * np.pi * kDerivativeCutoffFreq)
    controller = kc * (
        1
        + ctrl.tf([ki * DT, ki * DT], [2, -2], DT)
        + ctrl.tf([kd, -kd], [tau, -tau + DT], DT)
    )
    sign = -1.0 if negate_output else 1.0
    delay = ctrl.tf([1], np.append([1], np.zeros(delays + 1)), DT)
    return sign * controller * arx_transfer_function(NUM, DEN, DT) * delay


def response(sys, omega):
    return sys(np.exp(1j * omega * DT))


@pytest.mark.parametrize("delays", [1, 3])
@pytest.mark.parametrize("options", OPTIONS)
def test_loop_gain_is_independent_of_reference_path(delays, options):
    loop = LoopModel(NUM, DEN, DT, delays, GAINS, **options)
    expected = expected_loop_gain(
        delays, GAINS, negate_output=options.get("negate_output", False)
    )
    omega = np.geomspace(0.1, loop.nyquistFrequency(), 200)

    np.testing.assert_allclose(
        response(loop.loop_gain, omega), response(expected, omega), rtol=1e-6
    )


@pytest.mark.parametrize("options", OPTIONS)
def test_closed_loop_poles_are_the_roots_of_one_plus_loop_gain(options):
    loop = LoopModel(NUM, DEN, DT, 2, GAINS, **options)
    realization_poles = ctrl.poles(loop.closed_loop)

    for pole in loop.closedLoopPoles():
        assert np.min(np.abs(realization_poles - pole)) < 1e-4


def test_stability():
    assert LoopModel(NUM, DEN, DT, 1, GAINS).isStable()
    assert not LoopModel(NUM, DEN, DT, 1, UNSTABLE_GAINS).isStable()


def test_step_response_matches_reference_to_output_transfer_function():
    loop = LoopModel(NUM, DEN, DT, 2, GAINS)
    t = np.arange(0, 2.0, DT)
    y = loop.simulate(t, np.ones_like(t), np.zeros_like(t))
    _, y_expected = ctrl.step_response(loop.reference_to_output, T=t)

    np.testing.assert_allclose(y, y_expected, atol=1e-9)


def test_integral_action_removes_steady_state_error():
    loop = LoopModel(NUM, DEN, DT, 2, GAINS)
    t = np.arange(0, 60.0, DT)
    disturbance = np.full_like(t, -0.05)
    y = loop.simulate(t, np.ones_like(t), disturbance)

    assert y[-1] == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("options", OPTIONS)
def test_sensitivities(options):
    loop = LoopModel(NUM, DEN, DT, 2, GAINS, **options)
    omega = np.geomspace(0.1, loop.nyquistFrequency(), 200)
    gang = loop.sensitivities(omega)

    np.testing.assert_allclose(gang["S"] + gang["T"], 1.0, atol=1e-9)
    np.testing.assert_allclose(
        gang["TF"], response(loop.reference_to_output, omega), rtol=1e-6
    )


def test_peak_sensitivity_is_inverse_of_modulus_margin():
    loop = LoopModel(NUM, DEN, DT, 1, GAINS)
    omega = np.geomspace(1e-2, loop.nyquistFrequency(), 2000)
    peak_sensitivity = np.max(np.abs(loop.sensitivities(omega)["S"]))
    modulus_margin = loop.stabilityMargins()[2]

    assert peak_sensitivity == pytest.approx(1 / modulus_margin, rel=1e-3)


@pytest.mark.parametrize("delays", [1, 3])
def test_gain_margin_puts_closed_loop_poles_on_unit_circle(delays):
    loop = LoopModel(NUM, DEN, DT, delays, GAINS)
    gain_margin = loop.stabilityMargins()[0]

    assert np.max(np.abs(loop.closedLoopPoles(gain_margin))) == pytest.approx(
        1.0, abs=1e-6
    )


def test_root_locus_passes_through_current_closed_loop_poles():
    loop = LoopModel(NUM, DEN, DT, 2, GAINS)
    loci = loop.rootLocus(np.array([0.0, 1.0]))

    np.testing.assert_allclose(
        np.sort_complex(loci[0]), np.sort_complex(ctrl.poles(loop.loop_gain))
    )
    np.testing.assert_allclose(
        np.sort_complex(loci[1]), np.sort_complex(loop.closedLoopPoles()), atol=1e-9
    )


@pytest.mark.parametrize("delays", [1, 3])
def test_delay_margin_is_the_delay_that_destabilizes_the_loop(delays):
    loop = LoopModel(NUM, DEN, DT, delays, GAINS)
    _, phase_margin, _, _, gain_crossover, _ = loop.stabilityMargins()
    delay_margin, crossover = loop.delayMargin()

    assert delay_margin == pytest.approx(
        np.deg2rad(phase_margin) / gain_crossover, rel=1e-4
    )
    assert crossover == pytest.approx(gain_crossover, rel=1e-4)

    extra_samples = int(np.floor(delay_margin / DT))
    assert LoopModel(NUM, DEN, DT, delays + extra_samples, GAINS).isStable()
    assert not LoopModel(NUM, DEN, DT, delays + extra_samples + 1, GAINS).isStable()


def test_disk_margin_is_between_modulus_and_classical_margins():
    loop = LoopModel(NUM, DEN, DT, 1, GAINS)
    gain_margin, phase_margin, _, _, _, _ = loop.stabilityMargins()
    alpha, disk_gain_margin_db, disk_phase_margin_deg, _ = loop.diskMargins()

    assert 0 < alpha < 2
    assert disk_gain_margin_db < 20 * np.log10(gain_margin)
    assert disk_phase_margin_deg < phase_margin


@pytest.mark.parametrize("delays", [1, 3])
def test_loop_gain_touches_the_disk_margin_exclusion_disk(delays):
    loop = LoopModel(NUM, DEN, DT, delays, GAINS)
    alpha, _, _, _ = loop.diskMargins()
    a = (1 - alpha / 2) / (1 + alpha / 2)
    center = -(a + 1 / a) / 2
    radius = (1 / a - a) / 2
    omega = np.geomspace(1e-2, loop.nyquistFrequency(), 20000)
    distance_to_disk = np.abs(response(loop.loop_gain, omega) - center) - radius

    assert np.min(distance_to_disk) == pytest.approx(0.0, abs=1e-3)
