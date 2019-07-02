#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: VelocitySmoothing.py
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
"""

from numpy import *
import sys
import math
import matplotlib.pylab as plt

FLT_EPSILON = sys.float_info.epsilon
NAN = float('nan')
verbose = True;

if verbose:
    def verboseprint(*args):
        # Print each argument separately so caller doesn't need to
        # stuff everything to be printed into a single string
        for arg in args:
           print arg,
        print
else:
    verboseprint = lambda *a: None      # do-nothing function

class VelocitySmoothing(object):

    def setState(self, a0, v0, x0):
        self._accel = a0
        self._vel = v0
        self._pos = x0

    def __init__(self, dt, a0, v0, x0):
        self._jerk = 0.0
        self._accel = a0
        self._vel = v0
        self._pos = x0
        self._dt = dt
        self._vel_sp = 0.0

        self._max_jerk = 9.0
        self._max_accel = 3.0
        self._max_vel = 5.0

        self._max_jerk_T3 = 0.0

        self._T1 = 0.0
        self._T2 = 0.0
        self._T3 = 0.0

    def computeT1(self, a0, v3, j_max, a_max, dt):
        delta = 2.0*a0**2 + 4.0*j_max*v3

        if delta < 0.0:
            verboseprint('Complex roots\n')
            return (0.0, True);

        T1_plus = (-a0 + 0.5*sqrt(delta))/j_max
        T1_minus = (-a0 - 0.5*sqrt(delta))/j_max

        verboseprint('T1_plus = {}, T1_minus = {}'.format(T1_plus, T1_minus))
        # Use the solution that produces T1 >= 0 and T3 >= 0
        T3_plus = a0/j_max + T1_plus
        T3_minus = a0/j_max + T1_minus

        if T1_plus >= 0.0 and T3_plus >= 0.0:
            T1 = T1_plus
        elif T1_minus >= 0.0 and T3_minus >= 0.0:
            T1 = T1_minus
        else:
            verboseprint("Warning")
            T1 = 0.0

        if T1 < dt or T1 < 0.0:
            T1 = 0.0

        (T1, trapezoidal) = self.saturateT1ForAccel(a0, j_max, T1)

        return (T1, trapezoidal)

    def saturateT1ForAccel(self, a0, j_max, T1):
        trapezoidal = True
        # Check maximum acceleration, saturate and recompute T1 if needed
        a1 = a0 + j_max*T1
        if a1 > a_max:
            T1 = (a_max - a0) / j_max
            trapezoidal = True
        elif a1 < -a_max:
            T1 = (-a_max - a0) / j_max
            trapezoidal = True

        return (T1, trapezoidal)

    def computeT3(self, T1, a0, j_max, dt):
        T3 = a0/j_max + T1
        j_max_T3 = j_max

        if T1 < FLT_EPSILON and T3 < dt and T3 > FLT_EPSILON:
            # Force T3 to be the size of dt
            T3 = dt

            # Adjust new max jerk for adjusted T3
            # such that the acceleration can go from a0
            # to 0 in a single step (T3 = dt)
            j_max_T3 = a0/T3

        return (T3, j_max_T3)

    def computeT2(self, T1, T3, a0, v3, j_max):
        T2 = 0.0

        den = T1*j_max + a0
        if abs(den) > FLT_EPSILON:
            T2 = (-0.5*T1**2*j_max - T1*T3*j_max - T1*a0 + 0.5*T3**2*j_max - T3*a0 + v3)/den

        return T2

    def integrateT(self, dt, j, a_prev, v_prev, x_prev):
        a_T = j * dt + a_prev
        #v_T = j*dt*dt/2.0 + a_prev*dt + v_prev # Original equation: 3 mult + 1 div + 2 add
        v_T = dt/2.0 * (a_T + a_prev) + v_prev # Simplification using a_T: 1 mult + 1 div + 2 add
        #x_T = j*dt*dt*dt/6.0 + a_prev*dt*dt/2.0 + v_prev*dt + x_prev # Original equation: 6 mult + 2 div + 3 add
        x_T = dt/3.0 * (v_T + a_prev*dt/2.0 + 2*v_prev) + x_prev # Simplification using v_T: 3 mult + 2 div + 3 add

        return (a_T, v_T, x_T)

    def updateDurations(self):
        if (abs(self._accel) > self._max_accel):
            verboseprint("Should be double deceleration profile!")
        # Depending of the direction, start accelerating positively or negatively
        # For this, we need to predict what would be the velocity at zero acceleration
        # because it could be that the current acceleration is too high and that we need
        # to start reducing the acceleration directly even if sign(v_d - v_T) < 0
        if abs(self._accel) > FLT_EPSILON:
            j_zero_acc = -sign(self._accel) * abs(self._max_jerk);
            t_zero_acc = -self._accel / j_zero_acc;
            vel_zero_acc = self._vel + self._accel * t_zero_acc + 0.5 * j_zero_acc * t_zero_acc * t_zero_acc;
            verboseprint("vel_zero_acc = {}\tt_zero_acc = {}".format(vel_zero_acc, t_zero_acc))
            verboseprint("vel = {}, accel = {}, jerk = {}".format(self._vel, self._accel, j_zero_acc))
        else:
            vel_zero_acc = self._vel

        if self._vel_sp > vel_zero_acc:
            jerk = abs(self._max_jerk)
        else:
            jerk = -abs(self._max_jerk)

        delta_v = self._vel_sp - self._vel
        # Compute increasing acceleration time
        (T1, trapezoidal) = self.computeT1(self._accel, delta_v, jerk, self._max_accel, self._dt)
        # Compute decreasing acceleration time
	(T3, max_jerk_T3) = self.computeT3(T1, self._accel, jerk, self._dt);
        # Compute constant acceleration time
        if trapezoidal:
            T2 = self.computeT2(T1, T3, self._accel, delta_v, max_jerk_T3)
            if T2 < self._dt:
                verboseprint("T2 < dt: {}".format(T2))
                T2 = 0.0
        else:
            T2 = 0.0

        self._T1 = T1
        self._T2 = T2
        self._T3 = T3
        self._max_jerk_T3 = max_jerk_T3

    def update(self, dt, vel_sp):
        self._dt = dt
        self._vel_sp = clip(vel_sp, -self._max_vel, self._max_vel)
        self.updateDurations()

    def integrate(self, dt):
        if self._T1 > FLT_EPSILON:
            self._jerk = self._max_jerk_T3
        elif self._T2 > FLT_EPSILON:
            self._jerk = 0.0
        elif self._T3 > FLT_EPSILON:
            self._jerk = -self._max_jerk_T3
        else:
            self._jerk = 0.0

        # TODO: check new dt and compensate for jitter

        # Integrate trajectory
        (self._accel, self._vel, self._pos) = self.integrateT(self._dt, self._jerk, self._accel, self._vel, self._pos)

        return (self._accel, self._vel, self._pos)

    def computeEndPosition(self):
        j = self._max_jerk_T3
        T1 = self._T1
        T2 = self._T2
        T3 = self._T3
        xt3 = T1**3*j/6 + 0.5*T1**2*a0 + T1*v0 + T2**2*(0.5*T1*j + 0.5*a0) + T2*(0.5*T1**2*j + T1*a0 + v0) - T3**3*j/6 + T3**2*(0.5*T1*j + 0.5*a0) + T3*(0.5*T1**2*j + T1*a0 + T2*(T1*j + a0) + v0) + x0
        xt3_simp = T1*v0 + T3*(T1*a0 + T2*a0 + v0) + x0

        return (xt3, xt3_simp)


if __name__ == '__main__':
    # Initial conditions
    a0 = -8.0
    v0 = 12.0
    x0 = 0.0

    # Constraints
    j_max = 15.0
    a_max = 5.31
    v_max = 12.0

    # Simulation time parameters
    dt_0 = 0.014789
    t_end = 3.0

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
    v_d[0] = 0.0
    x_d = 30.0

    traj = VelocitySmoothing(dt_0, a0, v0, x0)
    traj._max_jerk = j_max
    traj._max_accel = a_max
    traj._max_vel = v_max
    traj._dt = dt_0
    braking_distance = VelocitySmoothing(dt_0, a0, v0, x0)
    braking_distance._max_jerk = j_max
    braking_distance._max_accel = a_max
    braking_distance._max_vel = v_max
    braking_distance._dt = dt_0

    xt3 = 0.0
    xt3_simp = 0.0

    # Main loop
    for k in range(0, n):
        verboseprint('k = {}\tt = {}'.format(k, t[k]))
        (a_T[k], v_T[k], x_T[k]) = traj.integrate(dt_0)
        traj.update(dt_0, v_d[k])
        if k == 0:
            # Predict the stop position if we decide to brake now
            braking_distance.setState(traj._accel, traj._vel, traj._pos)
            braking_distance.update(dt_0, 0.0)
            (xt3, xt3_simp) = braking_distance.computeEndPosition()

        verboseprint("T1 = {}\tT2 = {}\tT3 = {}\n".format(traj._T1, traj._T2, traj._T3))
        j_T[k] = traj._jerk

    print("Predicted final position = {} ; simplified: {}".format(xt3, xt3_simp))
    plt.plot(t, v_d)
    plt.plot(t, j_T, '*')
    plt.plot(t, a_T, '*')
    plt.plot(t, v_T)
    plt.plot(t, x_T)
    plt.plot(arange (0.0, t_end+dt_0, dt_0), t)
    plt.plot(t, j_T_corrected)
    plt.legend(["v_d", "j_T", "a_T", "v_T", "x_T", "t"])
    plt.xlabel("time (s)")
    plt.ylabel("metric amplitude")
    plt.show()
