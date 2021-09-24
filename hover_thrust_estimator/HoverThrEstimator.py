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
    The state is noise driven: Transition matrix A = 1
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

    def setMeasVarCoeff(self, coeff):
        self._R_gain = coeff

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
        self.setMeasVarCoeff(1.0)
        self.resetInnovSq()
        self.setInnovGateSize(3.0)
        self._dt = 1e-3
        self._noise_learning_time_constant = 2.0
        self._lpf_time_constant = 1.0
        self._residual_lpf = 0.0
        self._innov_test_ratio_signed_lpf = 0.0

    def predict(self, dt):
        # State is constant
        # Predict error covariance only
        self._P += self._Q * dt * dt
        self._dt = dt

    def fuseAccZ(self, acc_z, thrust):
        H = self.computeH(thrust)
        innov_var = self.computeInnovVar(H)
        K = self.computeKalmanGain(H, innov_var)
        innov = self.computeInnov(acc_z, thrust)
        innov_test_ratio = self.computeInnovTestRatio(innov, innov_var)

        residual = innov

        if self.isTestRatioPassing(innov_test_ratio):
            self.updateState(K, innov)
            self.updateStateCovariance(K, H)

            residual =  self.computeInnov(acc_z, thrust)

        elif not abs(self._innov_test_ratio_signed_lpf) < 0.2:
            self.bumpStateVariance()

        self.updateLpf(residual, sign(innov)*innov_test_ratio)
        self.updateMeasurementNoise(residual, H)
        return (innov, innov_var, innov_test_ratio)


    def computeH(self, thrust):
        return -9.81 * thrust / (self._hover_thr**2)

    def computeInnovVar(self, H):
        innov_var = H * self._P * H + (self._R * self._R_gain)
        return max(innov_var, self._R)

    def computeKalmanGain(self, H, innov_var):
        return self._P * H / innov_var

    def computeInnov(self, acc_z, thrust):
        return  acc_z - self.predictedAccZ(thrust)

    def predictedAccZ(self, thrust):
        return 9.81 * thrust / self._hover_thr - 9.81

    def computeInnovTestRatio(self, innov, innov_var):
        return innov**2 / (self._innov_gate_size**2 * innov_var)

    def isTestRatioPassing(self, innov_test_ratio):
        return (innov_test_ratio < 1.0)

    def updateState(self, K, innov):
        self._hover_thr += K * innov
        self._hover_thr = clip(self._hover_thr, 0.1, 0.9)

    def updateStateCovariance(self, K, H):
        self._P = clip((1.0 - K * H) * self._P, 1e-10, 1.0)

    def updateMeasurementNoise(self, residual, H):
        alpha = self._dt / (self._noise_learning_time_constant + self._dt)
        res_no_bias = residual - self._residual_lpf
        self._R = clip(self._R * (1.0 - alpha) + alpha * (res_no_bias**2 + H * self._P * H), 1.0, 400.0)

    def bumpStateVariance(self):
        self._P += 1000.0 * self._Q * self._dt

    def resetAccelNoise(self):
        self._R = 5.0

    def updateLpf(self, residual, innov_test_ratio_signed):
        alpha = self._dt / (self._lpf_time_constant + self._dt)
        self._residual_lpf = (1.0 - alpha) * self._residual_lpf + alpha * residual
        self._innov_test_ratio_signed_lpf = (1.0 - alpha) * self._innov_test_ratio_signed_lpf + alpha * clip(innov_test_ratio_signed, -1.0, 1.0)


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
