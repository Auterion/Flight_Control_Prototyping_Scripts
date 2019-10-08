#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: hover_thrust_sim.py
Author: Mathieu Bresciani
Email: brescianimathieu@gmail.com
Github: https://github.com/bresch
Description:
    Simple simulation of for the HoverThrEstimator class.
    A thrust curve is generated and produces acceleration
    using the same model as the estimator but with noise:

    accel = (Thrust - Hover thrust) * mass coefficient + noise
"""

from numpy import *
import matplotlib.pylab as plt
from HoverThrEstimator import HoverThrEstimator

def getAccelFromThrTime(thrust, t):
    accel = (thrust - hover_thr_true) * mass_coeff_true

    return accel

def getThrFromTime(t):
    if t < 0.6:
        thrust = t

    elif t < 1.5:
        thrust = 0.6

    elif t < 2.5:
        thrust = 0.5

    elif t < 4.0:
        thrust = 0.5 - (t - 2.5) * 0.25

    else:
        thrust = 0.0

    return thrust


if __name__ == '__main__':
    # Simulation parameters
    dt = 0.10
    t_end = 4.0
    t = arange (0.0, t_end+dt, dt)
    n = len(t)

    hover_thr_true = 0.6
    mass_coeff_true = 16.0

    # Estimator initial conditions
    hover_thr_0 = 0.5
    mass_coeff_0 = 20.0
    hover_thr_noise_0 = 0.2
    mass_coeff_noise_0 = 10.0
    hover_thr_process_noise = 0.02
    mass_coeff_process_noise = 1.0
    accel_noise = 3.5e-1

    hover_ekf = HoverThrEstimator(hover_thr_0, mass_coeff_0)
    hover_ekf.setStateVar(hover_thr_noise_0**2, mass_coeff_noise_0**2)
    hover_ekf.setProcessVar(hover_thr_process_noise**2, mass_coeff_process_noise**2)
    hover_ekf.setMeasVar(accel_noise**2)

    # Create data buckets
    accel = zeros(n)
    thrust = ones(n)
    hover_thr = zeros(n)
    mass_coeff = zeros(n)
    hover_thr_std = zeros(n)
    mass_coeff_std = zeros(n)

    for k in range(0, n):
        # Save data
        hover_thr[k] = hover_ekf._hover_thr
        mass_coeff[k] = hover_ekf._mass_coeff
        hover_thr_std[k] = sqrt(hover_ekf._P.item(0, 0))
        mass_coeff_std[k] = sqrt(hover_ekf._P.item(1, 1))

        # Generate measurement
        thrust[k] = getThrFromTime(t[k])
        noise = random.randn() * accel_noise
        accel[k] = getAccelFromThrTime(thrust[k], t[k]) + noise

        # Update the EKF
        hover_ekf.predict(dt)
        hover_ekf.fuseAccZ(accel[k], thrust[k])

    # Plot results
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(t, thrust * 10.0, '.')
    ax1.plot(t, accel, '.')
    ax1.legend(["Thrust (10x)", "AccZ"])

    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(t, hover_thr, 'b')
    ax2.plot(t, hover_thr_true * ones(n), 'k:')
    ax2.plot(t, hover_thr + hover_thr_std, 'g--')
    ax2.plot(t, hover_thr - hover_thr_std, 'g--')
    plt.legend(["Ht_Est", "Ht_True"])

    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(t, mass_coeff, 'r')
    ax3.plot(t, mass_coeff_true * ones(n), 'k:')
    ax3.plot(t, mass_coeff + mass_coeff_std, '--', color="tab:orange")
    ax3.plot(t, mass_coeff - mass_coeff_std, '--', color="tab:orange")
    plt.legend(["Cm_Est", "Cm_True"])
    plt.show()
