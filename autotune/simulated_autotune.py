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

File: simulated_autotune.py
Author: Mathieu Bresciani <mathieu@auterion.com>
License: BSD 3-Clause
Description:
    Auto-tuning algorithm test on simulated 2nd order transfer function
"""

import numpy as np
import matplotlib.pylab as plt
from scipy import signal
from arx_rls import ArxRls
from pid_design import computePidGmvc

def run():
    # Generate 2nd order transfer function
    zeta = 0.2 # damping ratio
    f_n = 2.0 # natural frequency
    w_n = f_n * 2.0*np.pi
    num = [w_n**2]
    den = [1, 2.0*zeta*w_n, w_n**2]
    Gs = signal.TransferFunction(num, den)

    # Simulation parameters
    n_steps = 1000
    t = np.linspace(0, 5, n_steps)
    u = np.ones(n_steps) # input signal
    u[int(n_steps/2):-1] = -1

    # Simulate the output of the continuous-time system
    t, y, x = signal.lsim(Gs, U=u, T=t)
    dt = t[1]

    # Identification
    n = 2 # order of the denominator (a_1,...,a_n)
    m = 2 # order of the numerator (b_0,...,b_m)
    d = 1
    rls = ArxRls(n, m, d)
    for k in range(n_steps):
        rls.update(u[k], y[k])
    theta_hat = rls._theta_hat

    # Construct discrete-time TF from vector of estimated parameters
    num = [theta_hat.item(i) for i in range(n, n+m+1)] # b0 .. bm
    den = [theta_hat.item(i) for i in range(0, n)] # a1 .. an
    den.insert(0, 1.0) # add 1 to get [1, a1, .., an]
    Gz = signal.TransferFunction(num, den, dt=dt)
    # TODO: add delay of d

    # Simulate the output and compare with the true system
    t, y_est = signal.dlsim(Gz, u, t=t)
    plt.plot(t, y, t, y_est)
    plt.legend(["True", "Estimated"])
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (-)")
    plt.show()

    # design controller
    (kc, ki, kd) = computePidGmvc(num, den, dt, sigma=0.1, delta=0.0, lbda=0.5)

    print("kc = {}, ki = {}, kd = {}\n".format(kc, ki, kd))
    return

if __name__ == '__main__':
    run()
