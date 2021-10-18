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

File: arx_rls.py
Author: Mathieu Bresciani <mathieu@auterion.com>
License: BSD 3-Clause
Description:
    Efficient recursive weighted least-squares algorithm
    without matrix inversion

    Assumes an ARX (autoregressive) model:
    A(q^-1)y(k) = q^-d * B(q^-1)u(k) + A(q^-1)e(k)

    with:
    q^-i  backward shift operator
    A(q^-1) = 1 + a_1*q^-1 +...+ a_n*q^-n
    B(q^-1) = b_0 + b_1*q^-1...+ b_m*q^-m
    n  order of A(q^-1)
    m  order of B(q^-1)
    d  delay
    u  input of the system
    y  output of the system
    e  white noise input

    usage:
    n = m = 2
    d = 1
    lbda = 0.95 # forgetting factor (0.9 < lbda < 0.99)
    rls = ArxRls(n, m, d, lbda)
    for i in range(0, n_steps):
        rls.update(u, y)
    theta_hat = rls._theta_hat

    theta_hat is of the form (a_1,..., a_n, b_0,..., b_n)

    References:
    - Identification de systemes dynamiques, D.Bonvin and A.Karimi, epfl, 2011
"""

import numpy as np

class ArxRls(object):
    def __init__(self, n, m, d, lbda=1.0):
        self._n = n
        self._m = m
        self._d = d
        p = n+m+1 # number of parameters (+1 because of b_0)
        self._P = 10000 * np.asmatrix(np.eye(p)) # initialize to a large value
        self._theta_hat = np.transpose(np.asmatrix(np.zeros(n+m+1)))
        self._y = np.zeros(n+1)
        self._u = np.zeros(m+d+1)
        self._lbda = lbda

    def update(self, u, y):
        self.addInputOutput(u, y)
        phi = self.constructPhi()
        lbda = self._lbda
        self._P = (self._P - self._P * phi * phi.T * self._P / (lbda + phi.T * self._P * phi))/lbda # eq 3.66
        self._theta_hat += self._P * phi*(self._y[self._n] - phi.T * self._theta_hat) # eq 3.67

        return self._theta_hat, self._P

    def addInputOutput(self, u, y):
        self.shiftRegisters()
        self._y[self._n] = y
        self._u[self._m+self._d] = u

    def shiftRegisters(self):
        for i in range(0, self._n):
            self._y[i] = self._y[i+1]
        for i in range(0, self._m+self._d):
            self._u[i] = self._u[i+1]

    def constructPhi(self):
        phi_a = np.asmatrix([-self._y[self._n-(i+1)] for i in range(self._n)])
        phi_b = np.asmatrix([self._u[self._m-i] for i in range(self._m+1)])
        phi = (np.concatenate((phi_a, phi_b), axis=1)).T
        return phi

if __name__ == '__main__':
    n = 2
    m = 1
    d = 1
    rls = ArxRls(n, m, d)

    assert len(rls._y) == n+1
    assert len(rls._u) == m+d+1
    rls.update(1, 2)
    rls.update(3, 4)
    rls.update(5, 6)
    assert rls._u.item(-1) == 5
    assert rls._y.item(-1) == 6
    phi = rls.constructPhi()
    assert len(phi) == 4
    assert  phi.item(0) == -4
    assert  phi.item(1) == -2
    assert  phi.item(2) == 3
    assert  phi.item(3) == 1
    print(rls._theta_hat)
