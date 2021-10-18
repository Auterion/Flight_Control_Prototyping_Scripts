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

import numpy as np
from arx_rls import ArxRls
from scipy.optimize import lsq_linear

class SystemIdentification(object):
    def __init__(self, n=2, m=2, d=1):
        self.n = n # order of the denominator (a_1,...,a_n)
        self.m = m # order of the numerator (b_0,...,b_m)
        self.d = d # number of delays
        self.forgetting_tc = 60.0 # forgetting factor for weighted RLS in seconds
        self.f_hp = 0.5 # high-pass filter cutoff frequency
        self.f_lp = 30.0 # low-pass filter cutoff frequency

    def run(self, t, u, y):
        n_steps = len(t)
        dt = t[1] - t[0]

        # High-pass filter parameters
        if self.f_hp > 0.0:
            tau_hp = 1/(2*np.pi*self.f_hp)
            alpha_hp = tau_hp/(tau_hp+dt)
        else:
            alpha_hp = 0.0

        u_hp = np.zeros(n_steps)
        y_hp = np.zeros(n_steps)
        u_hp[0] = u[0]
        y_hp[0] = y[0]

        # Low-pass filter parameters
        tau_lp = 1/(2*np.pi*self.f_lp)
        alpha_lp = tau_lp/(tau_lp+dt)
        u_lp = np.zeros(n_steps)
        y_lp = np.zeros(n_steps)
        u_lp[0] = u[0]
        y_lp[0] = y[0]

        a_coeffs = np.zeros((self.n, n_steps))
        b_coeffs = np.zeros((self.m + 1, n_steps))

        # Apply high and low-pass filters
        for k in range(n_steps):
            if k > 0:
                if alpha_hp > 0.0:
                    u_hp[k] = alpha_hp * u_hp[k-1] + alpha_hp*(u[k] - u[k-1])
                    y_hp[k] = alpha_hp * y_hp[k-1] + alpha_hp*(y[k] - y[k-1])
                else:
                    u_hp[k] = u[k]
                    y_hp[k] = y[k]

                u_lp[k] = alpha_lp * u_lp[k-1] + (1-alpha_lp)*u_hp[k]
                y_lp[k] = alpha_lp * y_lp[k-1] + (1-alpha_lp)*y_hp[k]


        use_rls = True
        if use_rls:
            # Identification
            rls = ArxRls(self.n, self.m, self.d, lbda=(1 - dt / self.forgetting_tc))

            for k in range(n_steps):
                # Update model
                rls.update(u_lp[k], y_lp[k])

                theta_hat = rls._theta_hat

                # Save for plotting
                for i in range(self.n):
                    a_coeffs[i,k] = theta_hat[i]
                for i in range(self.m+1):
                    b_coeffs[i,k] = theta_hat[i+self.n]

        else: # use LS
            # Build matrix of regressors
            A = np.zeros((n_steps, self.n+self.m+1))
            for row in range(n_steps):
                for i in range(self.n):
                    A[row,i] = -y_lp[row-(i+1)]
                for i in range(self.m+1):
                    A[row,i+self.n] = u_lp[row-(self.d+i)]

            B = [y_lp[i] for i in range(n_steps)] # Measured values

            res = lsq_linear(A, B, lsmr_tol='auto', verbose=1)
            theta_hat = res.x

            for i in range(self.n):
                a_coeffs[i,-1] = theta_hat[i]
            for i in range(self.m+1):
                b_coeffs[i,-1] = theta_hat[i+self.n]

        self.theta_hat = theta_hat

        return (theta_hat, a_coeffs, b_coeffs)

    def getNum(self):
        num = [self.theta_hat.item(i) for i in range(self.n, self.n+self.m+1)] # b0 .. bm
        return num

    def getDen(self):
        den = [self.theta_hat.item(i) for i in range(0, self.n)] # a1 .. an
        den.insert(0, 1.0) # add 1 to get [1, a1, .., an]
        return den
