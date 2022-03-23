#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
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
    rng_kin._vel_bottom_gate = 0.2

    n = len(t)
    inn = [0.0]
    rng_var = [0.0]
    test_ratio = [0.0]
    signed_test_ratio_lpf = [0.0]
    is_consistent = [False]

    for k in range(1, n):
        rng_var.append(0.1**2 + (0.05*rng[k])**2)
        rng_kin.update(rng[k], rng_var[k], vz[k], vz_var[k], t[k]*1e6)
        inn.append(rng_kin.getInnov())
        test_ratio.append(rng_kin._vel_bottom_test_ratio)
        signed_test_ratio_lpf.append(rng_kin.getSignedTestRatioLpf())
        is_consistent.append(rng_kin.isKinematicallyConsistent())

    plotData(t, rng, rng_var, is_consistent, vz, inn, test_ratio, signed_test_ratio_lpf)

def plotData(t, rng, rng_var, is_consistent, vz, inn, test_ratio, signed_test_ratio_lpf):
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
    ax3.plot(t, vz+inn)
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
