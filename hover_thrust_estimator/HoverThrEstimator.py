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

    def setInnovGateSize(self, gate_size):
        self._innov_gate_size = gate_size

    def __init__(self, hover_thr):
        self.setState(hover_thr)
        self.setStateVar(0.05)
        self.setProcessVar(0.3**2)
        self.setMeasVar(0.02)
        self.resetInnovSq()
        self.setInnovGateSize(3.0)
        self._predicted_acc_z = 0.0
        self._dt = 1e-3

    def predict(self, dt):
        # State is constant
        # Predict error covariance only
        self._P += self._Q * dt
        self._dt = dt

    def fuseAccZ(self, acc_z, thrust):
        H = self.computeH(thrust)
        acc_innov_var = H * self._P * H + self._R
        acc_innov_var = max(acc_innov_var, self._R)
        self._predicted_acc_z = self.predictedAccZ(thrust)
        acc_innov =  acc_z - self._predicted_acc_z

        kalman_gain = self._P * H / acc_innov_var

        # Compute test ratio for innovation filtering
        innov_test_ratio = acc_innov**2 / (self._innov_gate_size**2 * acc_innov_var)
        # TODO: use test ratio to filter the innovation and to bump the state covariance

        # Update state
        self._hover_thr += kalman_gain * acc_innov
        self._hover_thr = clip(self._hover_thr, 0.1, 0.8)

        residual =  acc_z - self.predictedAccZ(thrust)
        self.updateQR(residual, acc_innov, kalman_gain, H)

        # Update covariances
        self._P = max((1.0 - kalman_gain * H) * self._P, 0.0)

    def computeH(self, thrust):
        return -9.81 * thrust / (self._hover_thr**2)

    def predictedAccZ(self, thrust):
        return 9.81 * thrust / self._hover_thr - 9.81

    def updateQR(self, residual, innov, kalman_gain, H):
        tau = 0.5
        alpha = self._dt / (tau + self._dt)
        # self._Q = clip(self._Q * (1.0 - alpha) + alpha * (kalman_gain * innov**2 * kalman_gain), 1e-8, 0.1)
        self._R = clip(self._R * (1.0 - alpha) + alpha * (residual**2 + H * self._P * H), 1e-8, 10.0)


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
