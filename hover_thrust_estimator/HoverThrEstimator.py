#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: HoverThrEstimator.py
Author: Mathieu Bresciani
Email: brescianimathieu@gmail.com
Github: https://github.com/bresch
Description:
    1-state hover thrust estimator
    state: hover thrust (Th)
    The measurement is the vertical acceleration and the current
    thrust (T[k]) is used in the measurement model.
    The sate is noise driven: Transition matrix A = 1
    x[k+1] = Ax[k] + v with v ~ N(0, Q)
    y[k] = h(u, x) + w with w ~ N(0, R)
    Where the measurement model and corresponding Jocobian are:
    h(u, x) = g * T[k] / Th[k] - g
    H[k] = -g * T[k] / Th[k]**2
"""

from numpy import *
import sys
import math
import matplotlib.pylab as plt

innov_sq_length = 10
FLT_EPSILON = sys.float_info.epsilon
NAN = float('nan')
verbose = True;

if verbose:
    def verboseprint(*args):
        # Print each argument separately so caller doesn't need to
        # stuff everything to be printed into a single string
        for arg in args:
           print(arg)
        print()
else:
    verboseprint = lambda *a: None      # do-nothing function

class HoverThrEstimator(object):

    def setState(self, hover_thr):
        self._hover_thr = hover_thr

    def setStateVar(self, hover_thr_var):
        self._P = hover_thr_var

    def setProcessVar(self, hover_thr_process_noise_var):
        self._Q = hover_thr_process_noise_var

    def setMeasVar(self, accel_var):
        self._R = accel_var

    def resetInnovSq(self):
        self._innov_sq = 0.0
        self._C = 0.0
        self._nb_innov_sq = 0

    def __init__(self, hover_thr):
        self.setState(hover_thr)
        self.setStateVar(0.05)
        self.setProcessVar(0.3**2)
        self.setMeasVar(0.02)
        self.resetInnovSq()

    def predict(self, dt):
        # State is constant
        # Predict error covariance only
        self._P += self._Q * dt

    def fuseAccZ(self, acc_z, thrust):
        H = -9.81 * thrust / (self._hover_thr**2)
        acc_innov_var = H * self._P * H + self._R
        acc_innov_var = max(acc_innov_var, self._R)
        acc_innov = self.predictedAccZ(thrust) - acc_z
        self.addInnov(acc_innov)

        kalman_gain = self._P * H / acc_innov_var

        self.updateQR(kalman_gain, H)

        # Update state
        self._hover_thr -= kalman_gain * acc_innov
        self._hover_thr = clip(self._hover_thr, 0.1, 0.8)

        # Update covariances
        self._P = max((1.0 - kalman_gain * H) * self._P, 0.0)

    def addInnov(self, new_innov):
        self._innov_sq += new_innov**2

        self._nb_innov_sq += 1

    def predictedAccZ(self, thrust):
        return 9.81 * thrust / self._hover_thr - 9.81

    def updateQR(self, kalman_gain, H):
        if self._nb_innov_sq >= innov_sq_length:
            C = self._innov_sq / innov_sq_length
            self._Q = clip(kalman_gain * C * kalman_gain, 1e-8, 0.1)
            self._R = clip(C + H * self._P * H, 1e-8, 10.0)
            self._C = C
            print("C = ", C)

            self._nb_innov_sq = 0
            self._innov_sq = 0.0


if __name__ == '__main__':
    hover_thr_0 = 0.5
    hover_ekf = HoverThrEstimator(hover_thr_0)
    assert hover_ekf._hover_thr == hover_thr_0

    hover_thr_noise_0 = 0.2
    hover_ekf.setStateVar(hover_thr_noise_0**2)

    assert hover_ekf._P == hover_thr_noise_0**2

    hover_thr_process_noise = 0.01
    hover_ekf.setProcessVar(hover_thr_process_noise**2)
    assert hover_ekf._Q == hover_thr_process_noise**2

    dt = 0.01
    hover_ekf.predict(dt)
    assert hover_ekf._hover_thr == hover_thr_0
    assert hover_ekf._P == hover_thr_noise_0**2 + hover_thr_process_noise**2 * dt

    accel_noise = 0.1
    hover_ekf.setMeasVar(accel_noise**2)
    assert hover_ekf._R == accel_noise**2

    hover_ekf.fuseAccZ(0.0, hover_thr_0)
    assert hover_ekf._hover_thr == hover_thr_0
