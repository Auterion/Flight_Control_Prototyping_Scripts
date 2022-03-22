#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Copyright (c) 2022 PX4 Development Team
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

File: ConsistencyEstimator.py
Author: Mathieu Bresciani
Email: brescianimathieu@gmail.com
Github: https://github.com/bresch
Description:
"""

import numpy as np
import sys
import math
import matplotlib.pylab as plt
from AlphaFilter import AlphaFilter

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

class RangeFinderConsistencyCheck(object):
    def __init__(self):
        self._time_last_update_us = 0
        self._dist_bottom_prev = 0.0
        self._time_last_inconsistent_us = 0
        self._vel_bottom_gate = 0.1
        self._vel_bottom_signed_test_ratio_tau = 2.0
        self._consistency_hyst_time_us = 1.0
        self._min_vz_for_valid_consistency = 0.5
        self._vel_bottom_signed_test_ratio_lpf = AlphaFilter()
        self._is_kinematically_consistent = True

        self._vel_bottom_innov = 0.0
        self._vel_bottom_innov_var = 0.0
        self._vel_bottom_test_ratio = 0.0

    def update(self, dist_bottom, dist_bottom_var, vz, vz_var, time_us):
        dt = (time_us - self._time_last_update_us) * 1e-6

        if (self._time_last_update_us == 0) or (dt < 0.001) or (dt > 0.5):
            self._time_last_update_us = time_us
            self._dist_bottom_prev = dist_bottom
            return;

        vel_bottom = (dist_bottom - self._dist_bottom_prev) / dt
        innov = -vel_bottom - vz; # vel_bottom is +up while vz is +down
        vel_bottom_var = 2.0 * dist_bottom_var / (dt * dt) # Variance of the time derivative of a random variable: var(dz/dt) = 2*var(z) / dt^2
        innov_var = vel_bottom_var + vz_var
        normalized_innov_sq = (innov * innov) / innov_var
        self._vel_bottom_test_ratio = normalized_innov_sq / (self._vel_bottom_gate * self._vel_bottom_gate)

        self._vel_bottom_signed_test_ratio_lpf.setParameters(dt, self._vel_bottom_signed_test_ratio_tau)
        signed_test_ratio = np.sign(innov) * self._vel_bottom_test_ratio
        self._vel_bottom_signed_test_ratio_lpf.update(signed_test_ratio)

        self.updateConsistency(vz, time_us)

        self._time_last_update_us = time_us
        self._dist_bottom_prev = dist_bottom

        # Save for logging
        self._vel_bottom_innov = innov
        self._vel_bottom_innov_var = innov_var

    def updateConsistency(self, vz, time_us):
        if abs(self._vel_bottom_signed_test_ratio_lpf.getState()) >= 1.0:
            self._is_kinematically_consistent = False
            self._time_last_inconsistent_us = time_us

        else:
            if abs(vz) > self._min_vz_for_valid_consistency and self._vel_bottom_test_ratio < 1.0 and ((time_us - self._time_last_inconsistent_us) > self._consistency_hyst_time_us):
                self._is_kinematically_consistent = True

    def getInnov(self):
        return self._vel_bottom_innov

    def isKinematicallyConsistent(self):
        return self._is_kinematically_consistent

    def getSignedTestRatioLpf(self):
        return self._vel_bottom_signed_test_ratio_lpf.getState()

if __name__ == '__main__':
    rng_kin = RangeFinderConsistencyCheck()
    inn = []
    dt = 0.1

    for i in range(1, 100):
        rng_kin.update(i*dt, 0.1, -1.0, 0.001, i*dt * 1e6);
        inn.append(rng_kin.getInnov())

    import matplotlib.pylab as plt
    plt.plot(inn)
    plt.show()
