#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: velocity_trajectory_generator.py
Author: Mathieu Bresciani
Email: brescianimathieu@gmail.com
Github: https://github.com/bresch
Description:
    Given a desired velocity setpoint v_d, the trajectory generator computes
    a time-optimal trajectory satisfaying the following variable constraints:
    - j_max : maximum jerk
    - a_max : maximum acceleration
    - v_max : maximum velocity
    - a0 : initial acceleration
    - v0 : initial velocity
    - v3 : final velocity
    The hard constraint used to generate the optimizer is:
    - a3 = 0.0 : final acceleration

    The trajectory generated is made by three parts:
    1) Increasing acceleration during T1 seconds
    2) Constant acceleration during T2 seconds
    3) Decreasing acceleration during T3 seconds

    This script also generates a position setpoint which is computed
    as the integral of the velocity setpoint. If this one is bigger than
    x_err_max, the integration is frozen.

    The algorithm is simulated in a loop (typically that would be the position control loop)
    where the trajectory is recomputed a each iteration.
"""

from numpy import *
import matplotlib.pylab as plt


def integrate_T(j, a_prev, v_prev, x_prev, k, dt, a_max, v_max):

    a_T = j * dt + a_prev

    if a_T > a_max:
        a_T = a_max
    elif a_T < -a_max:
        a_T = -a_max

    v_T = dt/2.0 * (a_T + a_prev) + v_prev

    if v_T > v_max:
        v_T = v_max
    elif v_T < -v_max:
        v_T = -v_max

    x_T = dt/3.0 * (v_prev + a0*dt/2.0 + 2*v0) + x_prev

    return (a_T, v_T, x_T)


def compute_T1(a0, v0, v3, j_max, a_max):

    b = 2*a0/j_max
    c = v0/j_max + a0*a0/(2.0*j_max*j_max) - v3/j_max
    delta = b*b - 4.0*c
    if delta < 0.0:
        return 0.0
    T1_plus = (-b + sqrt(delta)) / 2.0
    T1_minus = (-b - sqrt(delta)) / 2.0

    T1 = max(max(T1_plus, T1_minus), 0.0)

    if T1 == 0.0:
        print("No feasible solution found, set T1 = 0")
        print("T1_plus = {} T1_minus = {}".format(T1_plus, T1_minus))

    # Check maximum acceleration, saturate and recompute T1 if needed
    a1 = a0 + j_max*T1
    if a1 > a_max:
        T1 = (a_max - a0) / j_max
    elif a1 < -a_max:
        T1 = (-a_max - a0) / j_max

    T1 = max(T1, 0.0)

    return T1

def compute_T3(T1, a0, j_max):
    T3 = a0/j_max + T1
    T3 = max(T3, 0.0)
    return T3

def compute_T2(T1, T3, a0, v0, v3, j_max):
    f = a0*T1 + j_max*T1*T1/2.0 + v0 + a0*T3 + j_max*T1*T3 - j_max*T3*T3/2.0
    T2 = (v3 - f) / (a0 + j_max*T1)
    T2 = max(T2, 0.0)
    return T2

# ============================================================
# ============================================================

# Initial conditions
a0 = 0.0
v0 = 0.0
x0 = 0.0

# Constraints
j_max = 22.1
a_max = 8.0
v_max = 6.0
x_err_max = 1.0

# Simulation time parameters
dt = 1.0/100.0
t_end = 10.0

# Initialize vectors
t = arange (0.0, t_end+dt, dt)
n = len(t)

j_T = zeros(n)
a_T = zeros(n)
v_T = zeros(n)
x_T = zeros(n)
v_d = zeros(n)

j_T[0] = 0.0
a_T[0] = a0
v_T[0] = v0
x_T[0] = x0
v_d[0] = -2.0

# Main loop
for k in range(0, n-1):

    # Change the desired velocity (simulate user RC sticks)
    if t[k] < 3.0:
        v_d[k+1] = v_d[k]
    elif t[k] < 4.0:
        v_d[k+1] = 4.0
    else:
        v_d[k+1] = -5.0

    # Depending of the direction, start accelerating positively or negatively
    if sign(v_d[k+1]-v_T[k]) > 0:
        j_max = abs(j_max)
    else:
        j_max = -abs(j_max)

    T1 = compute_T1(a_T[k], v_T[k], v_d[k+1], j_max, a_max)

    # Force T1/2/3 to zero if smaller than an epoch to avoid chattering
    if T1 < dt:
        T1 = 0.0

    T3 = compute_T3(T1, a_T[k], j_max)

    if T3 < dt:
        T3 = 0.0

    T2 = compute_T2(T1, T3, a_T[k], v_T[k], v_d[k+1], j_max)

    if T2 < dt:
        T2 = 0.0

    # Apply correct jerk (min, max or zero)
    if T1 > 0.0:
        j_T[k+1] = j_max
    elif T2 > 0.0:
        j_T[k+1] = 0.0
    elif T3 > 0.0:
        j_T[k+1] = -j_max
    else:
        j_T[k+1] = 0.0

    # Integrate the trajectory
    (a_T[k+1], v_T[k+1], x_T_new) = integrate_T(j_T[k+1], a_T[k], v_T[k], x_T[k], k, dt, a_max, v_max)

    # Lock the position setpoint if the error is bigger than some value
    drone_position = 0.0
    x_err = x_T_new - drone_position
    if abs(x_err) > x_err_max:
        x_T[k+1] = x_T[k]
    else :
        x_T[k+1] = x_T_new

# Plot trajectory and desired setpoint
plt.plot(t, v_d)
plt.plot(t, j_T)
plt.plot(t, a_T)
plt.plot(t, v_T)
plt.plot(t, x_T)
plt.legend(["v_d", "j_T", "a_T", "v_T", "x_T"])
plt.xlabel("time (s)")
plt.ylabel("metric amplitude")
plt.show()
