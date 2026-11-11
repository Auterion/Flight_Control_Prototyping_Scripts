"""Tests of the ARX identification.

Run with:  poetry run pytest test_system_identification.py
"""

import numpy as np
import pytest
from scipy.signal import lfilter
from system_identification import (
    SystemIdentification,
    arx_predict,
    measured_frequency_response,
    residual_correlations,
)


def first_order_response():
    # y(k) = 0.9 y(k-1) + 0.5 u(k-1) around a throttle of 0.5 and 5000 rpm
    rng = np.random.default_rng(0)
    u = rng.standard_normal(2000)
    y = np.zeros_like(u)
    for k in range(1, len(u)):
        y[k] = 0.9 * y[k - 1] + 0.5 * u[k - 1]
    return u + 0.5, y + 5000.0


def identify(u, y, method="OLS", remove_offset=True):
    sys_id = SystemIdentification(n=1, m=0, d=1, dt=0.01)
    sys_id.f_hp = 0.0
    sys_id.f_lp = 1000.0
    sys_id.remove_offset = remove_offset
    sys_id.fit(u, y, method=method)
    return sys_id


@pytest.mark.parametrize("method", ["OLS", "RLS"])
def test_operating_point_offset_does_not_bias_the_model(method):
    sys_id = identify(*first_order_response(), method=method)

    assert sys_id.getDen() == pytest.approx([1.0, -0.9], abs=0.02)
    assert sys_id.getNum() == pytest.approx([0.5], abs=0.05)


def test_offset_removal_can_be_disabled():
    sys_id = identify(*first_order_response(), remove_offset=False)

    # the offset is left for the ARX model to explain: a pole at z = 1
    assert sys_id.getDen()[1] == pytest.approx(-1.0, abs=0.01)


def test_one_step_prediction_is_the_arx_equation():
    num, den, delay = [0.5, 0.2], [1.0, -1.5, 0.6], 2
    rng = np.random.default_rng(1)
    u, y = rng.standard_normal(200), rng.standard_normal(200)

    prediction = arx_predict(num, den, delay, u, y, 1)

    k = np.arange(4, 200)
    expected = 1.5 * y[k - 1] - 0.6 * y[k - 2] + 0.5 * u[k - 2] + 0.2 * u[k - 3]
    assert prediction[k] == pytest.approx(expected)


def test_prediction_beyond_the_data_is_the_simulation():
    num, den, delay = [0.5, 0.2], [1.0, -1.5, 0.6], 2
    rng = np.random.default_rng(1)
    u, y = rng.standard_normal(200), rng.standard_normal(200)

    prediction = arx_predict(num, den, delay, u, y, 10 * len(u))

    simulation = lfilter([0.0, 0.0, 0.5, 0.2], den, u)
    assert prediction == pytest.approx(simulation)


DT = 0.01


def first_order_frequency_response(f):
    # y(k) = 0.9 y(k-1) + 0.1 u(k-1)
    z = np.exp(2j * np.pi * f * DT)
    return 0.1 / (z - 0.9)


def proportional_loop(k_p, noise_std):
    """Plant above under u = k_p (r - y_m), y_m measured with white noise."""
    rng = np.random.default_rng(2)
    r = rng.standard_normal(2**15)
    noise = noise_std * rng.standard_normal(len(r))
    y = np.zeros_like(r)
    u = np.zeros_like(r)
    for k in range(len(r)):
        if k > 0:
            y[k] = 0.9 * y[k - 1] + 0.1 * u[k - 1]
        u[k] = k_p * (r[k] - y[k] - noise[k])
    return r, u, y + noise


def relative_error(f, measured):
    expected = first_order_frequency_response(f)
    return np.abs(measured - expected) / np.abs(expected)


def bias(f, measured):
    # Averaging over frequencies leaves the bias, not the random error
    return np.abs(np.mean(measured / first_order_frequency_response(f)) - 1.0)


def test_measured_frequency_response_of_an_open_loop_is_the_plant():
    rng = np.random.default_rng(3)
    u = rng.standard_normal(2**13)
    y = lfilter([0.0, 0.1], [1.0, -0.9], u) + 0.001 * rng.standard_normal(len(u))

    f, measured, gamma2 = measured_frequency_response(u, y, DT)

    assert f[0] > 0.0
    assert np.all(gamma2 > 0.9)
    assert np.max(relative_error(f, measured)) < 0.1


def test_measured_frequency_response_from_the_reference_is_not_biased_by_feedback():
    r, u, y = proportional_loop(k_p=2.0, noise_std=0.5)

    f, from_input, _ = measured_frequency_response(u, y, DT)
    f, from_reference, gamma2 = measured_frequency_response(u, y, DT, r)

    is_good = gamma2 >= 0.6
    assert np.count_nonzero(is_good) > 10
    assert bias(f[is_good], from_reference[is_good]) < 0.05
    assert bias(f[is_good], from_input[is_good]) > 0.2


NUM, DEN, DELAY = [0.5, 0.2], [1.0, -1.5, 0.6], 2
MAX_LAG = 25


def arx_output(u, equation_error, delay=DELAY):
    forced = lfilter(np.concatenate((np.zeros(delay), NUM)), [1.0], u)
    return lfilter([1.0], DEN, forced + equation_error)


def lags_outside(correlation, bound):
    lags = np.arange(len(correlation)) - (len(correlation) - 1 - MAX_LAG)
    return lags[np.abs(correlation) > bound]


def test_residuals_of_the_true_model_are_white_and_uncorrelated_with_the_input():
    rng = np.random.default_rng(4)
    u = rng.standard_normal(4000)
    y = arx_output(u, 0.1 * rng.standard_normal(len(u)))

    auto, cross, bound = residual_correlations(NUM, DEN, DELAY, u, y, MAX_LAG)

    assert auto[0] == pytest.approx(1.0)
    assert len(lags_outside(auto[1:], bound)) <= 1
    assert len(lags_outside(cross, bound)) <= 1


def test_missed_dynamics_correlate_the_residuals_with_past_inputs():
    rng = np.random.default_rng(4)
    u = rng.standard_normal(4000)
    y = arx_output(u, 0.1 * rng.standard_normal(len(u)), delay=DELAY + 2)

    _, cross, bound = residual_correlations(NUM, DEN, DELAY, u, y, MAX_LAG)

    assert set(lags_outside(cross, bound)) >= {DELAY + 2, DELAY + 3}
    assert np.all(lags_outside(cross, bound) > 0)


def test_feedback_correlates_the_residuals_with_future_inputs_only():
    rng = np.random.default_rng(5)
    r = rng.standard_normal(4000)
    equation_error = rng.standard_normal(len(r))
    u = np.zeros_like(r)
    y = np.zeros_like(r)
    for k in range(len(r)):
        y[k] = equation_error[k]
        for i in range(1, len(DEN)):
            y[k] -= DEN[i] * y[k - i] if k >= i else 0.0
        for i, b in enumerate(NUM):
            y[k] += b * u[k - DELAY - i] if k >= DELAY + i else 0.0
        u[k] = 0.1 * (r[k] - y[k])

    _, cross, bound = residual_correlations(NUM, DEN, DELAY, u, y, MAX_LAG)

    assert 0 in lags_outside(cross, bound)
    assert np.all(lags_outside(cross, bound) <= 0)
