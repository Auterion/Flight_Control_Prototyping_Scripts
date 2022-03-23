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

File: range_finder_consistency_check.py
Author: Mathieu Bresciani
Email: brescianimathieu@gmail.com
Github: https://github.com/bresch
Description:
"""

import numpy as np
import matplotlib.pylab as plt
from data_extractor import getAllData
from RangeFinderConsistencyCheck import RangeFinderConsistencyCheck

def run(logfile):
    (t, rng, vz, vz_var) = getAllData(logfile)

    rng_kin = RangeFinderConsistencyCheck()
    rng_kin._vel_bottom_gate = 1.0

    n = len(t)
    inn = [0.0]
    rng_var = [0.0]
    test_ratio = [0.0]
    signed_test_ratio_lpf = [0.0]
    vel_bottom_var = [0.0]
    is_consistent = [False]

    for k in range(1, n):
        rng_var.append(0.01**2 + (0.01*rng[k])**2)
        rng_kin.update(rng[k], rng_var[k], vz[k], vz_var[k], t[k]*1e6)
        inn.append(rng_kin.getInnov())
        test_ratio.append(rng_kin._vel_bottom_test_ratio)
        signed_test_ratio_lpf.append(rng_kin.getSignedTestRatioLpf())
        is_consistent.append(rng_kin.isKinematicallyConsistent())
        vel_bottom_var.append(rng_kin._vel_bottom_innov_var - rng_var[k])

    plotData(t, rng, rng_var, is_consistent, vz, inn, test_ratio, signed_test_ratio_lpf, vel_bottom_var)

def plotData(t, rng, rng_var, is_consistent, vz, inn, test_ratio, signed_test_ratio_lpf, vel_bottom_var):
    n_plots = 4
    ax1 = plt.subplot(n_plots, 1, 1)
    ax1.plot(t, rng)
    ax1.plot(t, rng+np.sqrt(rng_var), 'r--')
    ax1.plot(t, rng-np.sqrt(rng_var), 'r--')
    ax1.legend(["rng"])
    ax1.grid()

    ax2 = plt.subplot(n_plots, 1, 2, sharex=ax1)
    ax2.plot(t, test_ratio)
    ax2.plot(t, signed_test_ratio_lpf)
    ax2.legend(["test_ratio", "signed_test_ratio_lpf"])
    ax2.set_ylim(-5, 5)
    ax2.grid()

    ax3 = plt.subplot(n_plots, 1, 3, sharex=ax1)
    ax3.plot(t, vz)
    rng_dot = vz+inn
    ax3.plot(t, rng_dot)
    ax3.plot(t, rng_dot+np.sqrt(vel_bottom_var), 'r--')
    ax3.plot(t, rng_dot-np.sqrt(vel_bottom_var), 'r--')
    ax3.set_ylim(-5, 5)
    ax3.legend(["vz", "rng_dot"])

    ax4 = plt.subplot(n_plots, 1, 4, sharex=ax1)
    ax4.plot(t, is_consistent)
    ax4.legend(["is_consistent"])

    plt.show()

if __name__ == '__main__':
    import os
    import argparse

    # Get the path of this script (without file name)
    script_path = os.path.split(os.path.realpath(__file__))[0]

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Estimate mag biases from ULog file')

    # Provide parameter file path and name
    parser.add_argument('logfile', help='Full ulog file path, name and extension', type=str)
    args = parser.parse_args()

    logfile = os.path.abspath(args.logfile) # Convert to absolute path

    run(logfile)
