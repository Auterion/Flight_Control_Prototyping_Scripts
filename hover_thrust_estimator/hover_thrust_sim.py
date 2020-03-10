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

def getAccelFromThrTime(thrust, t, ht):
    accel = 9.81 * thrust / ht - 9.81

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
        thrust = 0.8

    return thrust


if __name__ == '__main__':
    # Simulation parameters
    dt = 0.03
    t_end = 60.0
    t = arange (0.0, t_end+dt, dt)
    n = len(t)


    # Estimator initial conditions
    hover_thr_0 = 0.5
    hover_thr_noise_0 = 0.2
    P0 = hover_thr_noise_0**2
    hover_thr_process_noise = 0.01 # hover thrust change / s
    Q = hover_thr_process_noise**2
    Qk = Q * dt
    accel_noise = 3.5e-1 # in m/s2
    R = accel_noise**2 # Rk = R

    hover_ekf = HoverThrEstimator(hover_thr_0)
    hover_ekf.setStateVar(P0)
    hover_ekf.setProcessVar(Qk)
    hover_ekf.setMeasVar(R)

    # Create data buckets
    accel = zeros(n)
    accel_true = zeros(n)
    thrust = ones(n)
    hover_thr = zeros(n)
    hover_thr_std = zeros(n)
    hover_thr_true = zeros(n)
    accel_noise_estimated = zeros(n)

    for k in range(0, n):
        # Save data
        hover_thr[k] = hover_ekf._hover_thr
        hover_thr_std[k] = sqrt(hover_ekf._P)

        # Generate measurement
        thrust[k] = getThrFromTime(t[k])
        noise = random.randn() * accel_noise
        hover_thr_true[k] = 0.42 + 0.2 * (t[k]/t_end)
        accel_true[k] = getAccelFromThrTime(thrust[k], t[k], hover_thr_true[k])
        accel[k] =  accel_true[k] + noise

        # Accel noise change
        if t[k] > 40.0:
            accel_noise = 0.4
        elif t[k] > 30.0:
            accel_noise = 2.5

        # Update the EKF
        hover_ekf.predict(dt)
        hover_ekf.fuseAccZ(accel[k], thrust[k])
        accel_noise_estimated[k] = sqrt(hover_ekf._R)
        # print("P = ", hover_ekf._P, "\tQ = ", hover_ekf._Q, "\tsqrt(R) = ", sqrt(hover_ekf._R))

    # Plot results
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(t, thrust * 10.0, '.')
    ax1.plot(t, accel, '.')
    ax1.plot(t, accel_true + accel_noise_estimated, 'g--')
    ax1.plot(t, accel_true - accel_noise_estimated, 'g--')
    ax1.legend(["Thrust (10x)", "AccZ"])

    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.plot(t, hover_thr, 'b')
    ax2.plot(t, hover_thr_true, 'k:')
    ax2.plot(t, hover_thr + hover_thr_std, 'g--')
    ax2.plot(t, hover_thr - hover_thr_std, 'g--')
    plt.legend(["Ht_Est", "Ht_True"])

    plt.show()
