"""Tests of the ARX identification.

Run with:  poetry run pytest test_system_identification.py
"""

import numpy as np
import pytest
from system_identification import SystemIdentification


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
