"""Unit tests for the PID design (pid_design.py).

Run with:  poetry run pytest test_pid_design.py
"""

import numpy as np
import pytest
from loop_model import LoopModel
from pid_design import computePidGmvc, desiredClosedLoopPolynomial, solveDiophantine

DT = 0.008
DELAYS = 1
# Rate loop: 50 rad/s^2 per unit of torque, integrator and a 30 ms actuator lag
LAG_POLE = np.exp(-DT / 0.03)
DEN = list(np.convolve([1.0, -1.0], [1.0, -LAG_POLE]))
NUM = [50.0 * (1 - LAG_POLE) * DT]
MAX_OVERSHOOT = 0.1


def step_response(num, den, delays, gains, **options):
    t = np.arange(0.0, 3.0, DT)
    loop = LoopModel(num, den, DT, delays, gains, **options)
    return loop.simulate(t, np.ones_like(t), np.zeros_like(t))


def step_overshoot(num, den, delays, gains, **options):
    loop = LoopModel(num, den, DT, delays, gains, **options)
    if not loop.isStable():
        return np.inf
    return np.max(step_response(num, den, delays, gains, **options)) - 1.0


@pytest.fixture
def gmvc_gains():
    kc, ki, kd = computePidGmvc(NUM, DEN, DT, sigma=0.13, delta=0.0, lbda=0.5)
    return {"P": kc, "I": ki, "D": kd, "FF": 0.0}


def test_gmvc_overshoots_little_with_p_on_the_measurement(gmvc_gains):
    assert (
        step_overshoot(NUM, DEN, DELAYS, gmvc_gains, p_on_feedback=True) < MAX_OVERSHOOT
    )


def test_gmvc_overshoots_with_p_on_the_error(gmvc_gains):
    assert step_overshoot(NUM, DEN, DELAYS, gmvc_gains) > 2 * MAX_OVERSHOOT


def test_negative_feedforward_puts_p_on_the_measurement(gmvc_gains):
    p_on_error = {**gmvc_gains, "FF": -gmvc_gains["P"]}

    np.testing.assert_allclose(
        step_response(NUM, DEN, DELAYS, p_on_error),
        step_response(NUM, DEN, DELAYS, gmvc_gains, p_on_feedback=True),
        atol=1e-9,
    )


@pytest.mark.parametrize("delays", [0, 1, 4])
def test_diophantine_solution(delays):
    p = desiredClosedLoopPolynomial(DT, 0.13, 0.0)
    delta_a = np.convolve(DEN, [1.0, -1.0])
    e, f = solveDiophantine(p, delta_a, delays)

    rhs = np.convolve(delta_a, e)
    rhs[delays + 1 :] += f
    np.testing.assert_allclose(rhs, np.pad(p, (0, len(rhs) - len(p))), atol=1e-12)


def closed_form_gmvc(num, den, dt, sigma, delta, lbda):
    """GMVC gains for delays = 1, printed by pid_synthesis_symbolic.py."""
    a1, a2 = den[1], den[2]
    _, p1, p2 = desiredClosedLoopPolynomial(dt, sigma, delta)
    e1 = -a1 + p1 + 1
    f0 = -a1 * e1 + a1 - a2 + e1 + p2
    f1 = a1 * e1 - a2 * e1 + a2
    f2 = a2 * e1
    nu = lbda + (e1 + 1) * np.sum(num)
    return (
        -(f1 + 2 * f2) / nu,
        -(f0 + f1 + f2) / (dt * (f1 + 2 * f2)),
        -dt * f2 / (f1 + 2 * f2),
    )


@pytest.mark.parametrize(
    "num, den",
    [(NUM, DEN), ([0.167, -0.578, 0.651], [1.0, -1.689, 0.689])],
)
def test_gmvc_with_one_delay_is_the_closed_form(num, den):
    np.testing.assert_allclose(
        computePidGmvc(num, den, DT, 0.13, 0.5, 0.5, delays=1),
        closed_form_gmvc(num, den, DT, 0.13, 0.5, 0.5),
        rtol=1e-9,
    )


def design_overshoot(delays, design_delays):
    """Without detuning, lbda = 0: the plain GMVC design."""
    kc, ki, kd = computePidGmvc(NUM, DEN, DT, 0.13, 0.0, 0.0, delays=design_delays)
    gains = {"P": kc, "I": ki, "D": kd, "FF": 0.0}
    return step_overshoot(NUM, DEN, delays, gains, p_on_feedback=True)


@pytest.mark.parametrize("delays", [2, 4])
def test_gmvc_designed_for_the_plant_delay_keeps_the_loop_stable(delays):
    assert design_overshoot(delays, design_delays=1) == np.inf
    assert design_overshoot(delays, design_delays=delays) < MAX_OVERSHOOT


def gmvc_overshoot(num, den):
    kc, ki, kd = computePidGmvc(num, den, DT, 0.13, 0.0, 0.5)
    gains = {"P": kc, "I": ki, "D": kd, "FF": 0.0}
    return step_overshoot(num, den, DELAYS, gains, p_on_feedback=True)


def test_gmvc_of_a_first_order_model_is_a_pi():
    num, den = [50.0 * DT], [1.0, -1.0]
    _, _, kd = computePidGmvc(num, den, DT, 0.13, 0.0, 0.5)

    assert kd == 0.0
    assert gmvc_overshoot(num, den) < MAX_OVERSHOOT


def test_gmvc_uses_only_the_static_gain_of_the_numerator():
    spread_num = [0.5 * NUM[0], 0.3 * NUM[0], 0.1 * NUM[0], 0.1 * NUM[0]]

    np.testing.assert_allclose(
        computePidGmvc(spread_num, DEN, DT, 0.13, 0.0, 0.5),
        computePidGmvc(NUM, DEN, DT, 0.13, 0.0, 0.5),
    )


def test_gmvc_barely_changes_with_a_fast_third_pole():
    fast_pole = 0.05
    den = list(np.convolve(DEN, [1.0, -fast_pole]))
    num = [NUM[0] * (1 - fast_pole)]

    np.testing.assert_allclose(
        computePidGmvc(num, den, DT, 0.13, 0.0, 0.5),
        computePidGmvc(NUM, DEN, DT, 0.13, 0.0, 0.5),
        rtol=0.1,
    )
    assert gmvc_overshoot(num, den) < MAX_OVERSHOOT


def test_gmvc_derivative_gain_is_not_negative():
    # Poles of negative real part give a negative F second order term
    den = [1.0, -0.9, -0.1]

    assert computePidGmvc([0.1], den, DT, 0.13, 0.0, 0.5)[2] == 0.0
