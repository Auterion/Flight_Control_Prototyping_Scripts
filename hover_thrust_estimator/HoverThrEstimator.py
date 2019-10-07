#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: HoverThrEstimator.py
Author: Mathieu Bresciani
Email: brescianimathieu@gmail.com
Github: https://github.com/bresch
Description:
    2-state hover thrust estimator
    states: hover thrust (Th) and mass coefficient (Cm)
    The mass coefficient is used to convert from normalized thrust
    to acceleration.
    The measurement is the vertical acceleration and the current
    thrust (T[k]) is used in the measurement model.
    Both states are noise driven: Transition matrix A = eye(2)
    x[k+1] = Ax[k] + v with v ~ N(0, Q)
    y[k] = h(u, x) + w with w ~ N(0, R)
    Where the measurement model and corresponding Jocobian are:
    h(u, x) = (T[k] - Th[k]) * Cm[k]
    H[k] = [-Cm[k], T[k] - Th[k]]

    Observability:
    Cm is observable dynamically (T != Th)
    Th is observable statically (T == Th)
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

class HoverThrEstimator(object):

    def setState(self, hover_thr, mass_coeff):
        self._hover_thr = hover_thr
        self._mass_coeff = mass_coeff

    def setStateVar(self, hover_thr_var, mass_coeff_var):
        self._P = matrix([[hover_thr_var, 0.0],
                         [0.0, mass_coeff_var]])

    def setProcessVar(self, hover_thr_process_noise_var, mass_coeff_process_noise_var):
        self._Q = matrix([[hover_thr_process_noise_var, 0.0], [0.0, mass_coeff_process_noise_var]])

    def setMeasVar(self, accel_var):
        self._R = accel_var

    def __init__(self, hover_thr, mass_coeff):
        self.setState(hover_thr, mass_coeff)
        self.setStateVar(0.05, 101.0)
        self.setProcessVar(0.3**2, 1.0**2)
        self.setMeasVar(0.02)

    def predict(self, dt):
        # States are constant
        # Predict error covariance only
        self._P += self._Q * dt

    def fuseAccZ(self, acc_z, thrust):
        H = matrix([[-self._mass_coeff, thrust - self._hover_thr]])
        acc_innov_var = H * self._P * H.T + self._R
        acc_innov_var = max(acc_innov_var, self._R)
        acc_innov = self.predictedAccZ(thrust) - acc_z

        kalman_gain = self._P * H.T / acc_innov_var

        # Update states
        # TODO: hover_thr is by definition between 0 and 1
        # and it should always be between 0.15(racer) and 0.7(heavy)
        self._hover_thr -= kalman_gain.item(0) * acc_innov
        self._mass_coeff -= kalman_gain.item(1) * acc_innov

        # Update covariances
        self._P = maximum((eye(2) - kalman_gain * H) * self._P, self._P)

    def predictedAccZ(self, thrust):
        return thrust * self._mass_coeff - self._hover_thr * self._mass_coeff


if __name__ == '__main__':
    hover_thr_0 = 0.5
    mass_coeff_0 = 0.4
    hover_ekf = HoverThrEstimator(hover_thr_0, mass_coeff_0)
    assert hover_ekf._hover_thr == hover_thr_0
    assert hover_ekf._mass_coeff == mass_coeff_0

    hover_thr_noise_0 = 0.2
    mass_coeff_noise_0 = 10.0
    hover_ekf.setStateVar(hover_thr_noise_0**2, mass_coeff_noise_0**2)

    assert (hover_ekf._P == matrix([[hover_thr_noise_0**2, 0.0],
                                  [0.0, mass_coeff_noise_0**2]])).all()

    hover_thr_process_noise = 0.01
    mass_coeff_process_noise = 0.1
    hover_ekf.setProcessVar(hover_thr_process_noise**2, mass_coeff_process_noise**2)
    assert (hover_ekf._Q == matrix([[hover_thr_process_noise**2, 0.0],
                                    [0.0, mass_coeff_process_noise**2]])).all()

    dt = 0.01
    hover_ekf.predict(dt)
    assert hover_ekf._hover_thr == hover_thr_0
    assert hover_ekf._mass_coeff == mass_coeff_0
    assert (hover_ekf._P == matrix([[hover_thr_noise_0**2 + hover_thr_process_noise**2 * dt, 0.0],
                                  [0.0, mass_coeff_noise_0**2 + mass_coeff_process_noise**2 * dt]])).all()

    accel_noise = 0.1
    hover_ekf.setMeasVar(accel_noise**2)
    assert hover_ekf._R == accel_noise**2

    hover_ekf.fuseAccZ(0.0, hover_thr_0)
    assert hover_ekf._hover_thr == hover_thr_0
    assert hover_ekf._mass_coeff == mass_coeff_0
