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

def computePidGmvc(num, den, dt, sigma=0.1, delta=0.0, lbda=0.5):
    '''Compute a set of PID gains using General Minimum Variance Control law design

    Args:
        num: the coefficients of the numerator of the discrete-time plant transfer function
        den: the coefficients of the denominator of the discrete-time plant transfer function
        dt: the sampling time in seconds
        sigma: the desired closed-loop rise time in seconds
        delta: the damping index (between 0 and 2). 0 is critical damping, 1 is Butterworth
        lbda: the "detuning" coefficients. This affects the gain of the controller kc only. Increase to detune the controller.

    Raises:
        ValueError: if the is not a 2nd order system (2 poles)
        ValueError: if the plant has more than 2 zeros

    Returns:
        The PID gains in standard form: u = kc*[1 + ki*dt + kd/dt]*e
        kc: the controller gain
        ki: the integral gain (= 1/Ti)
        kd: the derivative gain (= Td)
    '''
    if len(den) != 3:
        print("Only supports 2nd order system")
        return(0.0, 0.0, 0.0)

    if len(num) > 3:
        print("Cannot have more than 2 zeros")
        return(0.0, 0.0, 0.0)

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

    # Solve GMVC law (see derivation in pid_synthesis_symbolic.py)
    rho = dt/sigma
    mu = 0.25 * (1-delta) + 0.51 * delta
    p1 = -2*np.exp(-rho/(2*mu))*np.cos(np.sqrt(4*mu-1)*rho/(2*mu))
    p2 = np.exp(-rho/mu)
    e1 = -a1 + p1 + 1
    f0 = -a1*e1 + a1 - a2 + e1 + p2
    f1 = a1*e1 - a2*e1 + a2
    f2 = a2*e1

    # Translate to PID gains
    nu = lbda + (e1 + 1)*(b0 + b1 + b2)
    kc = -(f1 + 2*f2)/nu
    ki = -(f0 + f1 + f2)/(dt*(f1 + 2*f2))
    kd = -dt*f2/(f1 + 2*f2)

    return (kc, ki, kd)

def gainsToNumDen(kc, ki, kd, dt):
    # use backwards Euler approximation for the derivative: s -> (z-1)/dt
    # and trapezoidal approximation for the integral: s -> 2/dt * (z-1)/(z+1)

    # Standard -> parallel form
    kI = kc * ki
    kD = kc * kd

    kIp = dt * kI / 2
    kDp = kD / dt

    b0 = kc + kIp + kDp
    b1 = -kc + kIp - 2*kDp
    b2 = kDp

    num = [b0, b1, b2]
    den = [1, -1, 0]
    return (num, den)

def computePidDahlin(num, den, dt, rise_time=1.0):
    if len(den) != 3:
        print("Only supports 2nd order system")
        return(0.0, 0.0, 0.0)

    if len(num) > 3:
        print("Cannot have zeros")
        return(0.0, 0.0, 0.0)


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
