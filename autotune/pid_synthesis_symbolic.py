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

File: pid_synthesis_symbolic.py
Author: Mathieu Bresciani <mathieu@auterion.com>
License: BSD 3-Clause
Description:
    Derivation of a PID controller using the relationship between
    PID and Generalized Minimum Variance Control (GMVC) law

    Reference:
    T.Yamatoto, K.Fujii and M.Kaneda, Design and implementation of a self-tuning pid controller, 1998
"""

from sympy import *
from sympy import linsolve

z = Symbol("z", real=True)

# Denominator coefficients 1 + a1*z^-1 + a2*z^-2
a1 = Symbol("a1", real=True)
a2 = Symbol("a2", real=True)

# Controller
f0 = Symbol("f0", real=True)
f1 = Symbol("f1", real=True)
f2 = Symbol("f2", real=True)
e1 = Symbol("e1", real=True)
e2 = Symbol("e2", real=True)

p1 = Symbol("p1", real=True)
p2 = Symbol("p2", real=True)

km = 2

P = 1 + p1 * z**-1 + p2 * z**-2
delta = 1 - z**-1
E = 1 + e1 * z**-1 + e2 * z**-2
F = f0 + f1 * z**-1 + f2 * z**-2
A = 1 + a1 * z**-1 + a2 * z**-2

# Compute eq.11 and collect the factors of each negative power of z
expanded = expand(delta * A * E + z**-(km +1) * F)
collected = collect(expanded, z)

# Compare with the the polynomial P to find e1 and f0
res_e1 = solveset(collected.coeff(z, -1) - p1, e1)
res_f0 = solveset(collected.coeff(z, -2) - p2, f0)

# Solve f1 and f2 given that the coefficients of
# z^-3 and z^-4 = 0
res_f1 = solveset(collected.coeff(z, -3), f1)
res_f2 = solveset(collected.coeff(z, -4), f2)

# Numerator coefficients b0 + b1*z^-1 + b2*z^-2
b0 = Symbol("b0", real=True)
b1 = Symbol("b1", real=True)
b2 = Symbol("b2", real=True)
lbda = Symbol("lbda", real=True)
B = b0 + b1 * z**-1 + b2 * z**-2
res_nu = E.subs(z, 1) * B.subs(z, 1) + lbda

# Compute kc, ki and kd from F and nu
kc = Symbol("kc", real=True)
KI = Symbol("KI", real=True)
KD = Symbol("KD", real=True)
dt = Symbol("dt", real=True)
nu = Symbol("nu", real=True)
# Parallel form is required to solve a linear set of equations
C = kc + KI*dt + KD/dt - (kc + 2*KD/dt) * z**-1 + KD/dt * z**-2
Eqns = [C.coeff(z, 0)-f0/nu, C.coeff(z, -1)-f1/nu, C.coeff(z, -2)-f2/nu]
res = linsolve(Eqns, kc, KI, KD)
# Transform results into standard form
res_kc = res.args[0][0]
res_ki = res.args[0][1]/res_kc
res_kd = res.args[0][2]/res_kc

print("# Parameters")
print("sigma = 0.04 # desired rise time in seconds")
print("delta = 1.0 # damping index, set between 0 and 2 (0 for binomial, 1 for Butterworth)")
print("lbda = 0.5 # tuning parameter, increase to increase robustness")
print("")
print("# Algorithm")
print("rho = dt/sigma")
print("mu = 0.25 * (1-delta) + 0.51 * delta")
print("p1 = -2*np.exp(-rho/(2*mu))*np.cos((np.sqrt(4*mu-1)*rho/(2*mu)))")
print("p2 = np.exp(-rho/mu)")
print("e1 = {}".format(res_e1.args[0]))
print("f0 = {}".format(res_f0.args[0]))
print("f1 = {}".format(res_f1.args[0]))
print("f2 = {}".format(res_f2.args[0]))
print("nu = {}".format(res_nu))
print("kc = {}".format(res_kc))
print("ki = {}".format(res_ki))
print("kd = {}".format(res_kd))
