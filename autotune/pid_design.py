#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Copyright (c) 2021 PX4 Development Team
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:

    1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.
    3. Neither the name PX4 nor the names of its contributors may be
    used to endorse or promote products derived from this software
    without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
    FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
    COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
    INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
    BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
    OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
    AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
    ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.

File: pid_design.py
Author: Mathieu Bresciani <mathieu@auterion.com>
License: BSD 3-Clause
Description:
    Design a PID controller based on the plant model
    and some design criteria
"""

import numpy as np
from numpy.polynomial import Polynomial


def desiredClosedLoopPolynomial(dt, sigma, delta):
    """P(z^-1) = 1 + p1 z^-1 + p2 z^-2 of the GMVC reference model.

    Discretization of 1 / (1 + sigma s + mu sigma^2 s^2), mu set by the
    damping index delta (Yamamoto, Fujii and Kaneda, 1998).
    """
    rho = dt / sigma
    mu = 0.25 * (1 - delta) + 0.51 * delta
    p1 = -2 * np.exp(-rho / (2 * mu)) * np.cos(np.sqrt(4 * mu - 1) * rho / (2 * mu))
    p2 = np.exp(-rho / mu)
    return np.array([1.0, p1, p2])


def solveDiophantine(p, delta_a, delays):
    """E and F of P = delta_a E + z^-(delays + 1) F, polynomials in z^-1.

    E (degree delays) is the start of the series P / delta_a, F the
    remainder, of degree len(delta_a) - 2.
    """
    remainder = np.zeros(len(delta_a) + delays)
    remainder[: len(p)] = p
    e = np.zeros(delays + 1)
    for i in range(delays + 1):
        e[i] = remainder[i] / delta_a[0]
        remainder[i : i + len(delta_a)] -= e[i] * delta_a
    return e, remainder[delays + 1 :]


def computePidGmvc(num, den, dt, sigma=0.1, delta=0.0, lbda=0.5, delays=1):
    """Compute a set of PID gains using General Minimum Variance Control law design

    Args:
        num: the coefficients of the numerator of the discrete-time plant transfer function
        den: the coefficients of the denominator of the discrete-time plant transfer function
        dt: the sampling time in seconds
        sigma: the desired closed-loop rise time in seconds
        delta: the damping index (between 0 and 2). 0 is critical damping, 1 is Butterworth
        lbda: the "detuning" coefficients. This affects the gain of the controller kc only. Increase to detune the controller.
        delays: the delay of the plant in samples, besides the sample of the
            measurement: the loop is y = z^-(delays + 1) num / den u

    The model may have any number of zeros, only its static gain is used,
    and at least one pole. A PID only realizes F up to the second order
    in (1 - z^-1): a first order model gives a PI controller, the higher
    orders of a model with more than two poles are dropped. A negative
    derivative gain, from poles of negative real part, is set to 0.

    Returns:
        The PID gains in standard form: u = kc*[1 + ki*dt + kd/dt]*e
        kc: the controller gain
        ki: the integral gain (= 1/Ti)
        kd: the derivative gain (= Td)
    """
    if len(den) < 2:
        print("Needs at least one pole")
        return (0.0, 0.0, 0.0)

    # Solve the GMVC law (see the derivation for delays = 1 in
    # pid_synthesis_symbolic.py)
    p = desiredClosedLoopPolynomial(dt, sigma, delta)
    e, f = solveDiophantine(p, np.convolve(den, [1.0, -1.0]), delays)

    # The velocity form PID is F / nu = kc * (dt / Ti + D + Td / dt * D^2),
    # D = 1 - z^-1: write F in powers of D, z^-1 = 1 - D
    f_in_differences = Polynomial(f)(Polynomial([1.0, -1.0])).coef
    integral, proportional, derivative = np.pad(f_in_differences, (0, 3))[:3]

    nu = lbda + np.sum(e) * np.sum(num)
    kc = proportional / nu
    ki = integral / (dt * proportional)
    kd = max(dt * derivative / proportional, 0.0)

    return (kc, ki, kd)


def gainsToNumDen(kc, ki, kd, dt):
    # use backwards Euler approximation for the derivative: s -> (z-1)/dt
    # and trapezoidal approximation for the integral: s -> 2/dt * (z-1)/(z+1)

    # Standard -> parallel form
    kI = kc * ki
    kD = kc * kd

    kIp = dt * kI / 2
    kDp = kD / dt

    b0 = kDp
    b1 = kc + kIp - 2 * kDp
    b2 = -kc + kIp + kDp

    num = [b0, b1, b2]
    den = [1, -1, 0]
    return (num, den)


def computePidDahlin(num, den, dt, rise_time=1.0):
    if len(den) != 3:
        print("Only supports 2nd order system")
        return (0.0, 0.0, 0.0)

    if len(num) > 3:
        print("Cannot have zeros")
        return (0.0, 0.0, 0.0)

    Q = 1.0 - np.exp(-dt / rise_time)
    a1 = den[1]
    a2 = den[2]
    b0 = num[0]

    if len(num) > 1:
        b1 = num[1]
    else:
        b1 = 0
    if len(num) > 2:
        b2 = num[2]
    else:
        b2 = 0
    b0 = b0 + b1 + b2

    kc = -(a1 + 2.0 * a2) * Q / b0

    Td = dt * a2 * Q / (kc * b0)
    Ti = -dt / (1.0 / (a1 + 2.0 * a2) + 1.0 + Td / dt)

    ki = 1.0 / Ti
    kd = Td

    return (kc, ki, kd)
