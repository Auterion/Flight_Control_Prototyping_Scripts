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

from __future__ import print_function

from numpy import *
import sys
import math
import matplotlib.pylab as plt

FLT_EPSILON = sys.float_info.epsilon
NAN = float('nan')
verbose = True

if verbose:
    def verboseprint(*args):
        # Print each argument separately so caller doesn't need to
        # stuff everything to be printed into a single string
        for arg in args:
            print(arg, end=" ")
        print("")
else:
    verboseprint = lambda *a: None      # do-nothing function

class VelocitySmoothing(object):

    def setState(self, a0, v0, x0):
        self._accel = a0
        self._vel = v0
        self._pos = x0

    def __init__(self, a0, v0, x0):
        self._jerk = 0.0
        self._accel = self._at0 = a0
        self._vel = self._vt0 = v0
        self._pos = self._xt0 = x0
        self._vel_sp = 0.0
        self._d = 0
        self._t0 = 0.0

        self._max_jerk = 9.0
        self._max_accel = 3.0
        self._max_vel = 5.0

        self._T1 = 0.0
        self._T2 = 0.0
        self._T3 = 0.0

    def computeT1(self, a0, v3, j_max, a_max):
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

        (T1, trapezoidal) = self.saturateT1ForAccel(a0, j_max, T1)

        return (T1, trapezoidal)

    def saturateT1ForAccel(self, a0, j_max, T1):
        trapezoidal = False
        # Check maximum acceleration, saturate and recompute T1 if needed
        a1 = a0 + j_max*T1
        if a1 > a_max:
            T1 = (a_max - a0) / j_max
            trapezoidal = True
        elif a1 < -a_max:
            T1 = (-a_max - a0) / j_max
            trapezoidal = True

        return (T1, trapezoidal)

    def computeT3(self, T1, a0, j_max):
        T3 = a0/j_max + T1

        return T3

    def computeT2(self, T1, T3, a0, v3, j_max):
        T2 = 0.0

        den = T1*j_max + a0
        if abs(den) > FLT_EPSILON:
            T2 = (-0.5*T1**2*j_max - T1*T3*j_max - T1*a0 + 0.5*T3**2*j_max - T3*a0 + v3)/den

        return T2

    def updateDurations(self, t):

        if (abs(self._accel) > self._max_accel):
            verboseprint("Should be double deceleration profile!")
        # Depending of the direction, start accelerating positively or negatively
        # For this, we need to predict what would be the velocity at zero acceleration
        # because it could be that the current acceleration is too high and that we need
        # to start reducing the acceleration directly even if sign(v_d - v_T) < 0
        do_stop_check = False
        if abs(self._accel) > FLT_EPSILON and do_stop_check:
            j_zero_acc = -sign(self._accel) * abs(self._max_jerk);
            t_zero_acc = -self._accel / j_zero_acc;
            vel_zero_acc = self._vel + self._accel * t_zero_acc + 0.5 * j_zero_acc * t_zero_acc * t_zero_acc;
            verboseprint("vel_zero_acc = {}\tt_zero_acc = {}".format(vel_zero_acc, t_zero_acc))
            verboseprint("vel = {}, accel = {}, jerk = {}".format(self._vel, self._accel, j_zero_acc))
        else:
            vel_zero_acc = self._vel

        self._d = sign(self._vel_sp - vel_zero_acc)
        jerk = self._d*self._max_jerk

        if abs(jerk) < 0.5 * self._max_jerk:
            self._T1 = self._T2 = self._T3 = 0.0
            print("Return")
            return

        self._at0 = self._accel
        self._vt0 = self._vel
        self._xt0 = self._pos

        delta_v = self._vel_sp - self._vel
        # Compute increasing acceleration time
        (T1, trapezoidal) = self.computeT1(self._accel, delta_v, jerk, self._max_accel)
        # Compute decreasing acceleration time
        T3 = self.computeT3(T1, self._accel, jerk);
        # Compute constant acceleration time
        if trapezoidal:
            T2 = self.computeT2(T1, T3, self._accel, delta_v, jerk)

        else:
            T2 = 0.0

        self._T1 = T1
        self._T2 = T2
        self._T3 = T3
        self._t0 = t

    def update(self, vel_sp, t):
        self._vel_sp = clip(vel_sp, -self._max_vel, self._max_vel)
        self.updateDurations(t)

    def evaluatePoly(self, j, a0, v0, x0, t, d):
        jt = d*j
        at = a0 + jt*t
        vt = v0 + a0*t + 0.5*jt*t**2
        xt = x0 + v0*t + 0.5*a0*t**2 + 1.0/6.0*jt*t**3

        return (jt, at, vt, xt)

    def evaluateTraj(self, t_now):
        """evaluate trajectory for the given time
        """
        j = self._max_jerk
        d = self._d
        t = t_now - self._t0

        if t <= self._T1:
            t1 = t
            (self._jerk, self._accel, self._vel, self._pos) = self.evaluatePoly(j, self._at0, self._vt0, self._xt0, t1, d)

        elif t <= self._T1 + self._T2:
            t1 = self._T1
            t2 = t - self._T1

            (self._jerk, self._accel, self._vel, self._pos) = self.evaluatePoly(j, self._at0, self._vt0, self._xt0, t1, d)
            (self._jerk, self._accel, self._vel, self._pos) = self.evaluatePoly(0.0, self._accel, self._vel, self._pos, t2, 0.0)

        elif t <= self._T1 + self._T2 + self._T3:
            t1 = self._T1
            t2 = self._T2
            t3 = t - self._T1 - self._T2

            (self._jerk, self._accel, self._vel, self._pos) = self.evaluatePoly(j, self._at0, self._vt0, self._xt0, t1, d)
            (self._jerk, self._accel, self._vel, self._pos) = self.evaluatePoly(0.0, self._accel, self._vel, self._pos, t2, 0.0)
            (self._jerk, self._accel, self._vel, self._pos) = self.evaluatePoly(j, self._accel, self._vel, self._pos, t3, -d)

        else:
            # TODO : do not recompute if the sequence has already been completed
            t1 = self._T1
            t2 = self._T2
            t3 = self._T3

            (self._jerk, self._accel, self._vel, self._pos) = self.evaluatePoly(j, self._at0, self._vt0, self._xt0, t1, d)
            (self._jerk, self._accel, self._vel, self._pos) = self.evaluatePoly(0.0, self._accel, self._vel, self._pos, t2, 0.0)
            (self._jerk, self._accel, self._vel, self._pos) = self.evaluatePoly(j, self._accel, self._vel, self._pos, t3, -d)
            (self._jerk, self._accel, self._vel, self._pos) = self.evaluatePoly(0.0, 0.0, self._vel_sp, self._pos, t - self._T1 - self._T2 - self._T3, 0.0)

        return (self._jerk, self._accel, self._vel, self._pos)


if __name__ == '__main__':
    # Initial conditions
    a0 = 1.18
    v0 = 2.52
    x0 = 0.0

    # Constraints
    j_max = 8.0
    a_max = 4.0
    v_max = 12.0

    # Simulation time parameters
    dt_0 = 0.02
    t_end = 6.0

    # Initialize vectors
    t = arange (0.0, t_end+dt_0, dt_0)
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
    v_d[0] = 2.34

    traj = VelocitySmoothing(a0, v0, x0)
    traj._max_jerk = j_max
    traj._max_accel = a_max
    traj._max_vel = v_max
    traj._dt = dt_0

    # Main loop
    for k in range(0, n):
        if k != 0:
            if t[k] < 1.0:
                v_d[k] = v_d[k-1]
            elif t[k] < 1.3:
                v_d[k] = 1.0
            elif t[k] < 3.0:
                v_d[k] = 1.5
            else:
                v_d[k] = -3.0
        verboseprint('k = {}\tt = {}'.format(k, t[k]))
        (j_T[k], a_T[k], v_T[k], x_T[k]) = traj.evaluateTraj(t[k])
        traj.update(v_d[k], t[k])

        verboseprint("T1 = {}\tT2 = {}\tT3 = {}\n".format(traj._T1, traj._T2, traj._T3))

    plt.plot(t, v_d)
    plt.plot(t, j_T)
    plt.plot(t, a_T)
    plt.plot(t, v_T)
    plt.plot(t, x_T, '--')
    plt.legend(["v_d", "j_T", "a_T", "v_T", "x_T"])
    plt.xlabel("time (s)")
    plt.ylabel("metric amplitude")
    plt.show()
