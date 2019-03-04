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
import sys
import math

FLT_EPSILON = sys.float_info.epsilon
NAN = float('nan')


def integrate_T(j, a_prev, v_prev, x_prev, dt, a_max, v_max):

    a_T = j * dt + a_prev


    #v_T = j*dt*dt/2.0 + a_prev*dt + v_prev # Original equation: 3 mult + 1 div + 2 add
    v_T = dt/2.0 * (a_T + a_prev) + v_prev # Simplification using a_T: 1 mult + 1 div + 2 add

    #x_T = j*dt*dt*dt/6.0 + a_prev*dt*dt/2.0 + v_prev*dt + x_prev # Original equation: 6 mult + 2 div + 3 add
    x_T = dt/3.0 * (v_T + a_prev*dt/2.0 + 2*v_prev) + x_prev # Simplification using v_T: 3 mult + 2 div + 3 add

    return (a_T, v_T, x_T)

def recomputeMaxJerk(a0, v3, T1, j_max):

    # If T1 is smaller than dt, it means that the jerk is too large to reach the
    # desired acceleration with a bang-bang signal => recompute the maximum jerk

    print(a0, v3, T1)
    delta = 2.0*T1**2*a0**2 - 4.0*T1*a0*v3 + v3**2
    if delta < 0.0:
        return 0.0
    j_new_minus = -0.5*(2.0*T1*a0 - v3)/T1**2 - 0.5*sqrt(delta)/T1**2
    j_new_plus = -0.5*(2.0*T1*a0 - v3)/T1**2 + 0.5*sqrt(delta)/T1**2
    print('j_new_plus = {}, j_new_minus = {}'.format(j_new_plus, j_new_minus))
    (T1_plus, T3_plus) = compute_T1_T3(a0, v3, j_new_plus)
    (T1_minus, T3_minus) = compute_T1_T3(a0, v3, j_new_minus)
    if T1_plus >= 0.0 and T3_plus >= 0.0:
        j_new = j_new_plus
        if T1_minus >= 0.0 and T3_minus >= 0.0:
            print('Both jerks are valid; check for time optimality')
            if T1_plus + T3_plus > T1_minus + T3_minus:
                j_new = j_new_minus
    elif T1_minus >= 0.0 and T3_minus >= 0.0:
        j_new = j_new_minus
        if T1_plus >= 0.0 and T3_plus >= 0.0:
            print('Both jerks are valid; check for optimality')
            if T1_plus + T3_plus < T1_minus + T3_minus:
                j_new = j_new_plus
    else:
        print('Error in recomputeMaxJerk')
        j_new = j_max

    return j_new

def compute_T1_T3(a0, v3, j_max):

    delta = 2.0*a0**2 + 4.0*j_max*v3
    if delta < 0.0:
        print('Complex roots\n')
        return (0.0, j_max);

    T1_plus = (-a0 + 0.5*sqrt(delta))/j_max
    T1_minus = (-a0 - 0.5*sqrt(delta))/j_max

    print('T1_plus = {}, T1_minus = {}'.format(T1_plus, T1_minus))
    # Use the solution that produces T1 >= 0 and T3 >= 0
    T3_plus = a0/j_max + T1_plus
    T3_minus = a0/j_max + T1_minus

    if T1_plus >= 0.0 and T3_plus >= 0.0:
        T1 = T1_plus
        T3 = T3_plus
    elif T1_minus >= 0.0 and T3_minus >= 0.0:
        T1 = T1_minus
        T3 = T3_minus
    else:
        T1 = 0.0
        T3 = NAN

    return (T1, T3)

def compute_T1(a0, v3, j_max, a_max, dt):

    (T1, T3) = compute_T1_T3(a0, v3, j_max)
    j_max_T1 = j_max

    if T1 < dt or T1 < 0.0:
        return (0.0, j_max)

    # Check maximum acceleration, saturate and recompute T1 if needed
    a1 = a0 + j_max_T1*T1
    if a1 > a_max:
        T1 = (a_max - a0) / j_max_T1
    elif a1 < -a_max:
        T1 = (-a_max - a0) / j_max_T1

    if j_max_T1 == 0.0:
        T1 = 0.0
        j_max_T1 = j_max

    return (T1, j_max_T1)


def computeT1_T123(T123, accel_prev, vel_prev, vel_setpoint, max_jerk):
	a = -max_jerk;
	b = max_jerk * T123 - accel_prev;
        c = vel_prev + accel_prev * T123 - accel_prev**2 / (2.0 * max_jerk) - vel_setpoint
	delta = b**2 - 4.0 * a * c

	sqrt_delta = sqrt(delta);
	denominator_inv = 1.0 / (2.0 * a);
	T1_plus = (-b + sqrt_delta) * denominator_inv;
	T1_minus = (-b - sqrt_delta) * denominator_inv;
        T1_plus = max(T1_plus, 0.0)
        T1_minus = max(T1_minus, 0.0)
        (T3_plus, j_max_T3) = compute_T3(T1_plus, accel_prev, vel_prev, vel_setpoint, max_jerk)
        (T3_minus, j_max_T3) = compute_T3(T1_minus, accel_prev, vel_prev, vel_setpoint, max_jerk)
        if (T1_plus + T3_plus > T123):
            T1 = T1_minus
        elif (T1_minus + T3_minus > T123):
            T1 = T1_plus
        else:
            T1 = max(T1_plus, T1_minus)

        print("plus = {}, minus = {}".format(T1_plus, T1_minus))

	return T1

def compute_T3(T1, a0, v3, j_max, dt):
    T3 = a0/j_max + T1
    j_max_T3 = j_max

    if T1 == 0.0 and T3 < dt and T3 > 0.0:
        # Adjust T3 to be a multiple of dt
        print('Exact T3 = {}'.format(T3))
        T3 = dt
        print('Rounded T3 = {}'.format(T3))

        # Adjust new max jerk for adjusted T3
        j_max_T3 = a0/T3
        print('Full jerk = {}'.format(j_max))
        if abs(j_max_T3) > abs(j_max):
            j_max_T3 = j_max

        print('T3 New jerk = {}\n'.format(j_max_T3))

    T3 = max(T3, 0.0)
    return (T3, j_max_T3)

def compute_T2(T1, T3, a0, v3, j_max, dt):
    T2 = 0.0

    if abs(T1*j_max + a0) > FLT_EPSILON:
        T2 = (-0.5*T1**2*j_max - T1*T3*j_max - T1*a0 + 0.5*T3**2*j_max - T3*a0 + v3)/(T1*j_max + a0)

    if T2 < dt:
        T2 = 0.0

    T2 = max(T2, 0.0)
    return T2

def compute_T2_T123(T123, T1, T3):
    T2 = T123 - T1 - T3
    return T2

# ============================================================
# ============================================================

# Initial conditions
a0 = 0.0
v0 = 0.5
x0 = 0.0

# Constraints
j_max = 9.0
a_max = 6.0
v_max = 6.0

# Simulation time parameters
dt_0 = 1.0/50.0
t_end = 5.2

# Initialize vectors
t = arange (0.0, t_end+dt_0, dt_0)
n = len(t)

j_T = zeros(n)
j_T_corrected = zeros(n)
a_T = zeros(n)
v_T = zeros(n)
x_T = zeros(n)
v_d = zeros(n)

j_T[0] = 0.0
j_T_corrected[0] = 0.0
a_T[0] = a0
v_T[0] = v0
x_T[0] = x0
v_d[0] = -2.0

print_T123 = True

dt_prev = dt_0
sigma_jitter = 0.0
sigma_jitter = dt_0/5.0

# Main loop
for k in range(0, n-1):
    dt = dt_0 + random.randn() * sigma_jitter # Add jitter

    if k > 0:
        t[k] = t[k-1] + dt
        print('k = {}\tt = {}'.format(k, t[k]))

        # Correct the jerk if dt is bigger than before and that we only need one step of jerk to complete phase T1 or T3
        # This helps to avoid overshooting and chattering around zero acceleration due to dt jitter
        if dt > dt_prev \
                and ( \
                        (dt > T1 and T1 > FLT_EPSILON) \
                        or (dt > T3 and T3 > FLT_EPSILON)):
            j_T_corrected[k-1] = j_T[k-1] * dt_prev / dt
        else:
            j_T_corrected[k-1] = j_T[k-1]

        # Integrate the trajectory
        (a_T[k], v_T[k], x_T[k]) = integrate_T(j_T_corrected[k-1], a_T[k-1], v_T[k-1], x_T[k-1], dt, a_max, v_max)

    # Change the desired velocity (simulate user RC sticks)
    if t[k] < 3.0:
        v_d[k] = v_d[0]
    elif t[k] < 4.5:
        v_d[k] = 4.0
    else:
        v_d[k] = 5.0

    # Depending of the direction, start accelerating positively or negatively
    if sign(v_d[k]-v_T[k]) > 0:
        j_max = abs(j_max)
    else:
        j_max = -abs(j_max)

    (T1, j_max_T1) = compute_T1(a_T[k], v_d[k] - v_T[k], j_max, a_max, dt)

    (T3, j_max_T1) = compute_T3(T1, a_T[k], v_d[k] - v_T[k], j_max_T1, dt)

    T2 = compute_T2(T1, T3, a_T[k], v_d[k] - v_T[k], j_max_T1, dt)


    # Apply correct jerk (min, max or zero)
    if T1 > 0.0:
        j_T[k] = j_max_T1
    elif T2 > 0.0:
        j_T[k] = 0.0
    elif T3 > 0.0:
        j_T[k] = -j_max_T1
    else:
        j_T[k] = 0.0
        print('T123 = 0, t = {}\n'.format(t[k]))
        print("T1 = {}\tT2 = {}\tT3 = {}\n".format(T1, T2, T3))

# end loop


print('=========== END ===========')
# Plot trajectory and desired setpoint
plt.plot(t, v_d)
plt.plot(t, j_T)
plt.plot(t, a_T)
plt.plot(t, v_T)
plt.plot(t, x_T)
plt.step(arange (0.0, t_end+dt_0, dt_0), t)
plt.legend(["v_d", "j_T", "a_T", "v_T", "x_T", "t"])
plt.xlabel("time (s)")
plt.ylabel("metric amplitude")
plt.show()
