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
    where the mass coefficient is g / Hover thrust
    which gives
    accel = g * Thrust / Hover thrust - g
"""

from numpy import *
import matplotlib.pylab as plt
from HoverThrEstimator import HoverThrEstimator

def getAccelFromThrTime(thrust, t):
    accel = 9.81 * thrust / hover_thr_true - 9.81

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

    elif t < 6.0:
        thrust = 0.42
    else:
        thrust = 0.0

    return thrust


if __name__ == '__main__':
    # Simulation parameters
    dt = 0.03
    t_end = 8.0
    t = arange (0.0, t_end+dt, dt)
    n = len(t)

    hover_thr_true = 0.42

    # Estimator initial conditions
    hover_thr_0 = 0.5
    hover_thr_noise_0 = 0.2
    hover_thr_process_noise = 0.01
    accel_noise = 3.5e-1

    hover_ekf = HoverThrEstimator(hover_thr_0)
    hover_ekf.setStateVar(hover_thr_noise_0**2)
    hover_ekf.setProcessVar(hover_thr_process_noise**2)
    hover_ekf.setMeasVar((10.0 * accel_noise)**2)

    # Create data buckets
    accel = zeros(n)
    thrust = ones(n)
    hover_thr = zeros(n)
    hover_thr_std = zeros(n)

    for k in range(0, n):
        # Save data
        hover_thr[k] = hover_ekf._hover_thr
        hover_thr_std[k] = sqrt(hover_ekf._P)

        # Generate measurement
        thrust[k] = getThrFromTime(t[k])
        noise = random.randn() * accel_noise
        if t[k] > 1.5:
            hover_thr_true = 0.6
        accel[k] = getAccelFromThrTime(thrust[k], t[k]) + noise

        # Update the EKF
        hover_ekf.predict(dt)
        hover_ekf.fuseAccZ(accel[k], thrust[k])
        print("P = ", hover_ekf._P, "\tQ = ", hover_ekf._Q, "\tR = ", hover_ekf._R)

    # Plot results
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(t, thrust * 10.0, '.')
    ax1.plot(t, accel, '.')
    ax1.legend(["Thrust (10x)", "AccZ"])

    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.plot(t, hover_thr, 'b')
    ax2.plot(t, hover_thr_true * ones(n), 'k:')
    ax2.plot(t, hover_thr + hover_thr_std, 'g--')
    ax2.plot(t, hover_thr - hover_thr_std, 'g--')
    plt.legend(["Ht_Est", "Ht_True"])

    plt.show()
