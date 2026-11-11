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

File: system_identification.py
Author: Mathieu Bresciani <mathieu@auterion.com>
License: BSD 3-Clause
Description:
    Class performing input and output signal pre-processing (e.g.: remove bias and
    unwanted frequencies using high and low-pass filters) and
    running a weighted RLS algorithm to identify an ARX parametric model
"""

import control as ctrl
import numpy as np
from arx_rls import ArxRls
from scipy.optimize import lsq_linear
from scipy.signal import coherence, csd, lfilter


def apply_filters(u, y, f_hp, f_lp, dt):
    n_steps = len(u)

    if f_hp > 0.0:
        tau_hp = 1 / (2 * np.pi * f_hp)
        alpha_hp = tau_hp / (tau_hp + dt)
    else:
        alpha_hp = 0.0

    tau_lp = 1 / (2 * np.pi * f_lp)
    alpha_lp = tau_lp / (tau_lp + dt)

    u_hp = np.zeros(n_steps)
    y_hp = np.zeros(n_steps)
    u_hp[0] = u[0]
    y_hp[0] = y[0]
    u_lp = np.zeros(n_steps)
    y_lp = np.zeros(n_steps)
    u_lp[0] = u[0]
    y_lp[0] = y[0]

    for k in range(1, n_steps):
        if alpha_hp > 0.0:
            u_hp[k] = alpha_hp * u_hp[k - 1] + alpha_hp * (u[k] - u[k - 1])
            y_hp[k] = alpha_hp * y_hp[k - 1] + alpha_hp * (y[k] - y[k - 1])
        else:
            u_hp[k] = u[k]
            y_hp[k] = y[k]

        u_lp[k] = alpha_lp * u_lp[k - 1] + (1 - alpha_lp) * u_hp[k]
        y_lp[k] = alpha_lp * y_lp[k - 1] + (1 - alpha_lp) * y_hp[k]

    return u_lp, y_lp


def arx_transfer_function(num, den, dt, **kwargs):
    # num and den hold coefficients in powers of z^-1 (b0..bm, 1..an) while
    # ctrl.TransferFunction expects powers of z: right-pad num to the order of
    # den, otherwise n - m spurious samples of delay are added.
    num = np.concatenate((num, np.zeros(max(len(den) - len(num), 0))))
    return ctrl.TransferFunction(num, den, dt, **kwargs)


def arx_predict(num, den, delay, u, y, horizon):
    """Output of the ARX model predicted `horizon` samples ahead.

    y(k) is predicted from the measured outputs up to k - horizon and the
    measured inputs: a horizon of 1 is the one-step predictor the model is fitted
    with, a horizon longer than the data the simulation of the model.
    """
    forced = lfilter(np.concatenate((np.zeros(delay), num)), [1.0], u)
    measured = np.asarray(y, dtype=float)
    order = len(den) - 1
    recent = []  # predictions at horizons h - 1, h - 2, ... (at most order)
    for h in range(1, min(horizon, len(u)) + 1):
        y_h = forced.copy()
        for i in range(1, order + 1):
            past = recent[i - 1] if i <= len(recent) else measured
            y_h[i:] -= den[i] * past[:-i]
        recent = [y_h] + recent[: order - 1]
    return recent[0] if recent else measured


def measured_frequency_response(u, y, dt, r=None):
    """Frequency response from u to y estimated from the data by Welch averages.

    Returns (f, H, coherence) without the zero frequency. In closed loop,
    Φ_uy / Φ_uu tends to -1/C where the noise fed back by the controller
    dominates the excitation: given the reference r carrying the excitation,
    H = Φ_ry / Φ_ru is not biased by that noise. Its coherence is the lower one
    of r to u and r to y.
    """
    fs = 1.0 / dt
    nperseg = min(1024, len(u) // 4)
    source = u if r is None else r
    f, p_su = csd(source, u, fs, nperseg=nperseg)
    _, p_sy = csd(source, y, fs, nperseg=nperseg)
    if r is None:
        _, gamma2 = coherence(u, y, fs, nperseg=nperseg)
    else:
        _, gamma2_ru = coherence(r, u, fs, nperseg=nperseg)
        _, gamma2_ry = coherence(r, y, fs, nperseg=nperseg)
        gamma2 = np.minimum(gamma2_ru, gamma2_ry)
    return f[1:], p_sy[1:] / p_su[1:], gamma2[1:]


def residual_correlations(num, den, delay, u, y, max_lag):
    """Residual tests of an ARX model (Ljung, System Identification, 1999).

    The residuals are the one-step prediction errors of the model on u and y,
    the data it was fitted to. Returns (auto, cross, bound):
    - auto: autocorrelation of the residuals at lags 0 to max_lag, all within
      the bound if the residuals are white, as least squares assumes;
    - cross: correlation of the residual e(t) with the input u(t - lag) at lags
      -max_lag to max_lag. At positive lags it reveals dynamics the model
      misses; at lag 0 and negative lags, an input reacting to the current
      and past residuals, as a controller in closed loop does;
    - bound: 99 % confidence bound of these correlations for white residuals.
    """
    start = max(len(den) - 1, delay + len(num) - 1)
    residuals = (y - arx_predict(num, den, delay, u, y, 1))[start:]
    u = u[start:]

    def correlation(a, b, lag):
        n = len(a)
        if lag >= 0:
            return np.dot(a[lag:], b[: n - lag]) / n
        return np.dot(a[: n + lag], b[-lag:]) / n

    e_power = correlation(residuals, residuals, 0)
    auto = np.array(
        [correlation(residuals, residuals, lag) for lag in range(max_lag + 1)]
    )
    cross = np.array(
        [correlation(residuals, u, lag) for lag in range(-max_lag, max_lag + 1)]
    )
    auto /= e_power
    cross /= np.sqrt(e_power * correlation(u, u, 0))
    return auto, cross, 2.58 / np.sqrt(len(residuals))


class SysIdResult(object):
    def __init__(self, num, den, dt):
        self.G_ = arx_transfer_function(num, den, dt)


class SystemIdentification(object):
    def __init__(self, n=2, m=2, d=1, dt=0.0):
        self.n = n  # order of the denominator (a_1,...,a_n)
        self.m = m  # order of the numerator (b_0,...,b_m)
        self.d = d  # number of delays
        self.forgetting_tc = 60.0  # forgetting factor for weighted RLS in seconds
        self.f_hp = 0.5  # high-pass filter cutoff frequency
        self.f_lp = 30.0  # low-pass filter cutoff frequency
        self.remove_offset = True
        self.dt = dt

        tau = 60.0  # forgetting period
        self.lbda = 1.0 - self.dt / tau

    def fit(self, u, y, method="RLS"):
        n_steps = len(u)

        a_coeffs = np.zeros((self.n, n_steps))
        b_coeffs = np.zeros((self.m + 1, n_steps))

        u_lp, y_lp = self.preprocess(u, y)

        if method == "RLS":
            rls = ArxRls(
                self.n, self.m, self.d, lbda=(1 - self.dt / self.forgetting_tc)
            )
            for k in range(n_steps):
                rls.update(u_lp[k], y_lp[k])
                theta_hat = rls._theta_hat
                for i in range(self.n):
                    a_coeffs[i, k] = theta_hat[i]
                for i in range(self.m + 1):
                    b_coeffs[i, k] = theta_hat[i + self.n]
        elif method == "OLS":
            skip = max(self.n, self.d + self.m)
            rows = n_steps - skip
            A = np.zeros((rows, self.n + self.m + 1))
            B = np.zeros(rows)
            for row in range(skip, n_steps):
                for i in range(self.n):
                    A[row - skip, i] = -y_lp[row - (i + 1)]
                for i in range(self.m + 1):
                    A[row - skip, i + self.n] = u_lp[row - (self.d + i)]
                B[row - skip] = y_lp[row]
            res = lsq_linear(A, B, lsmr_tol="auto")
            theta_hat = res.x
            for i in range(self.n):
                a_coeffs[i, -1] = theta_hat[i]
            for i in range(self.m + 1):
                b_coeffs[i, -1] = theta_hat[i + self.n]

        self.theta_hat = theta_hat

        estimate = SysIdResult(self.getNum(), self.getDen(), self.dt)
        return estimate

    def preprocess(self, u, y):
        """Input and output the model is fitted to."""
        if self.remove_offset:
            # The ARX model has no constant term: an operating point offset
            # would be fitted as a pole at z = 1. Removing the means is the
            # least-squares equivalent of SIDPAC's bias regressor (Morelli,
            # AIAA 2002-4704)
            u, y = u - np.mean(u), y - np.mean(y)
        return apply_filters(u, y, self.f_hp, self.f_lp, self.dt)

    def getNum(self):
        num = [
            self.theta_hat.item(i) for i in range(self.n, self.n + self.m + 1)
        ]  # b0 .. bm
        return num

    def getDen(self):
        den = [self.theta_hat.item(i) for i in range(0, self.n)]  # a1 .. an
        den.insert(0, 1.0)  # add 1 to get [1, a1, .., an]
        return den
