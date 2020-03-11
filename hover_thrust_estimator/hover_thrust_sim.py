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
    if t <= 10.0:
        thrust = 0.8
    else:
        thrust = 0.5

    return thrust

def getHoverThrustFromTime(t):
    if t <= 10.0:
        hover_t = 0.8
    else:
        hover_t = 0.5

    return hover_t


if __name__ == '__main__':
    # Simulation parameters
    dt = 0.03
    t_end = 60.0
    t = arange (0.0, t_end+dt, dt)
    n = len(t)


    # Estimator initial conditions
    hover_thrust_0 = 0.5
    hover_thrust_noise_0 = 0.1
    P0 = hover_thrust_noise_0**2
    hover_thrust_process_noise = sqrt(0.25e-6) # hover thrust change / s
    Q = hover_thrust_process_noise**2
    Qk = Q
    accel_noise_0 = sqrt(5.0)
    R = accel_noise_0**2 # Rk = R

    hover_ekf = HoverThrEstimator(hover_thrust_0)
    hover_ekf.setStateVar(P0)
    hover_ekf.setProcessVar(Qk)
    hover_ekf.setMeasVar(R)

    accel_noise_sim = 2.0 # in m/s2
    hover_thrust_sim = 0.8

    # Create data buckets
    accel = zeros(n)
    accel_true = zeros(n)
    thrust = ones(n)
    hover_thrust = zeros(n)
    hover_thrust_std = zeros(n)
    hover_thrust_true = zeros(n)
    accel_noise_std = zeros(n)
    innov_test_ratio_lpf = zeros(n)
    innov = zeros(n)
    innov_std = zeros(n)
    innov_test_ratio = zeros(n)
    residual_lpf = zeros(n)

    for k in range(0, n):
        # Save data
        hover_thrust[k] = hover_ekf._hover_thr
        hover_thrust_std[k] = sqrt(hover_ekf._P)

        # Generate measurement
        thrust[k] = getThrFromTime(t[k])
        noise = random.randn() * accel_noise_sim
        hover_thrust_true[k] = getHoverThrustFromTime(t[k])
        accel_true[k] = getAccelFromThrTime(thrust[k], t[k], hover_thrust_true[k])
        accel[k] =  accel_true[k] + noise

        # Accel noise change
        if t[k] > 40.0:
            accel_noise_sim = 2.0
        elif t[k] > 30.0:
            accel_noise_sim = 4.0

        # Update the EKF
        hover_ekf.predict(dt)
        (innov[k], innov_var, innov_test_ratio[k]) = hover_ekf.fuseAccZ(accel[k], thrust[k])
        innov_std[k] = sqrt(innov_var)
        accel_noise_std[k] = sqrt(hover_ekf._R)
        innov_test_ratio_lpf[k] = hover_ekf._innov_test_ratio_lpf
        residual_lpf[k] = hover_ekf._residual_lpf
        # print("P = ", hover_ekf._P, "\tQ = ", hover_ekf._Q, "\tsqrt(R) = ", sqrt(hover_ekf._R))

    # Plot results
    n_plots = 5
    ax1 = plt.subplot(n_plots, 1, 1)
    ax1.plot(t, thrust * 10.0, '.')
    ax1.plot(t, accel, '.')
    ax1.plot(t, accel_true + 3*accel_noise_std, 'g--')
    ax1.plot(t, accel_true - 3*accel_noise_std, 'g--')
    ax1.legend(["Thrust (10x)", "AccZ"])

    ax2 = plt.subplot(n_plots, 1, 2, sharex=ax1)
    ax2.plot(t, hover_thrust, 'b')
    ax2.plot(t, hover_thrust_true, 'k:')
    ax2.plot(t, hover_thrust + hover_thrust_std, 'g--')
    ax2.plot(t, hover_thrust - hover_thrust_std, 'g--')
    ax2.legend(["Ht_Est", "Ht_True"])

    ax3 = plt.subplot(n_plots, 1, 3, sharex=ax1)
    ax3.plot(t, hover_thrust_std)
    ax3.legend(["Ht_noise"])

    ax4 = plt.subplot(n_plots, 1, 4, sharex=ax1)
    ax4.plot(t, innov_test_ratio)
    ax4.plot(t, innov_test_ratio_lpf)
    ax4.plot(t, residual_lpf)
    ax4.legend(["Test ratio", "Test ratio lpf", "Residual lpf"])

    ax5 = plt.subplot(n_plots, 1, 5, sharex=ax1)
    ax5.plot(t, innov)
    ax5.plot(t, innov + innov_std, 'g--')
    ax5.plot(t, innov - innov_std, 'g--')
    ax5.legend(["Innov"])

    plt.show()
