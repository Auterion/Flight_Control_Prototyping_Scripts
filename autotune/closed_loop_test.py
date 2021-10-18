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

File: autotune_sim.py
Author: Mathieu Bresciani <mathieu@auterion.com>
License: BSD 3-Clause
Description:
    Simulate closed-loop system with generated PID controller
"""
import numpy as np
import control as ctrl
import matplotlib.pylab as plt
from pid_design import computePidGmvc, gainsToNumDen

# Discrete-time model given by system identification
## jMavsim
# num = [0.167, -0.578, 0.651]
# den = [ 1.        , -1.689,  0.689]
# Gazebo standard vtol
num = [0.086, -0.255, 0.231]
den = [ 1.        , -1.864,  0.864]
dt = 0.0036

# 2nd order model
# zeta = 1
# f_n = 2.0
# w_n = 2*np.pi*f_n
# pa1 = -zeta*w_n+w_n*np.sqrt((zeta**2-1)+0j)
# pa2 = -zeta*w_n-w_n*np.sqrt((zeta**2-1)+0j)
# p1 = np.exp(pa1*dt)
# p2 = np.exp(pa2*dt)
# num = [1, 2, 1]
# den = [1, -(p1+p2), p1*p2]
# K = sum(den)/sum(num)*0.5
# num = [K, 2*K, K]

tau_ol = dt / (1.0 - den[2])
print("Tau open-loop = {}\n".format(tau_ol))

# sigma = 0.1 # rise time
sigma = 3 * tau_ol # rise time
delta = 0.0 # damping property, set between 0 and 2 (1 for Butterworth)
lbda = 0.5
(kc, ki, kd) = computePidGmvc(num, den, dt, sigma, delta, lbda)
ki = ki / 5
print("Standard: kc = {}, ki = {}, kd = {}\n".format(kc, ki, kd))
print("Parallel: kp = {}, ki = {}, kd = {}\n".format(kc, kc*ki, kc*kd))

# Compute reference model (just for plotting)
rho = dt/sigma
mu = 0.25 * (1-delta) + 0.51 * delta
p1 = -2*np.exp(-rho/(2*mu))*np.cos((np.sqrt(4*mu-1)*rho/(2*mu)))
p2 = np.exp(-rho/mu)
P = ctrl.TransferFunction([1+p1+p2],[1, p1, p2], dt)

Gz2 = ctrl.TransferFunction(num, den, dt)
(pid_num, pid_den) = gainsToNumDen(kc, ki, kd, dt)
PID = ctrl.TransferFunction(pid_num, pid_den, dt)
Gcl = ctrl.feedback(PID * Gz2, 1)

# Simulate step response
t_ref, y_ref = ctrl.step_response(P, T=np.arange(0,1,dt))
t_out, y_out = ctrl.step_response(Gcl, T=np.arange(0,1,dt))

plt.plot(t_out, y_out)
plt.plot(t_ref, y_ref)
plt.title("Closed-loop step response")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (rad/s)")
plt.legend(["Closed-loop", "Reference model"])
plt.show()
