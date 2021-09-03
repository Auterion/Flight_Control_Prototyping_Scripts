#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: hover_thrust_replay.py
Author: Mathieu Bresciani
Email: brescianimathieu@gmail.com
Github: https://github.com/bresch
Description:
"""

from numpy import *
import matplotlib.pylab as plt
from HoverThrEstimator import HoverThrEstimator
from pyulog import ULog
import os


def get_data(log, topic_name, variable_name):
    for elem in log.data_list:
        if elem.name == topic_name:
            variable_data = elem.data[variable_name]
            break

    return variable_data

def ms2s_list(time_ms_list):
    return [i*1e-6 for i in time_ms_list]

def run(log_name, showplots):
    log = ULog(log_name)

    # Select msgs and copy into arrays
    thrust = -get_data(log, 'vehicle_local_position_setpoint', 'thrust[2]')
    az = -get_data(log, 'vehicle_local_position', 'az')
    vx = get_data(log, 'vehicle_local_position', 'vx')
    vy = get_data(log, 'vehicle_local_position', 'vy')
    vz = -get_data(log, 'vehicle_local_position', 'vz')
    dist_bottom = get_data(log, 'vehicle_local_position', 'dist_bottom')
    t = ms2s_list(get_data(log, 'vehicle_local_position_setpoint', 'timestamp'))
    t_ekf = ms2s_list(get_data(log, 'vehicle_local_position', 'timestamp'))

    # Downsample ekf estimate to setpoint sample rate
    accel = array(interp(t, t_ekf, az))
    vel_x = array(interp(t, t_ekf, vx))
    vel_y = array(interp(t, t_ekf, vy))
    vel_z = array(interp(t, t_ekf, vz))

    # Estimator initial conditions
    hover_thrust_0 = 0.5
    hover_thrust_noise_0 = 0.1
    P0 = hover_thrust_noise_0**2
    hover_thrust_process_noise = 0.0036 # hover thrust change / s
    Q = hover_thrust_process_noise**2
    accel_noise_0 = sqrt(5.0)
    R = accel_noise_0**2 # Rk = R

    # Speed sensitivity reduction
    vz_thr = 2.0
    vxy_thr = 10.0

    hover_ekf = HoverThrEstimator(hover_thrust_0)
    hover_ekf.setStateVar(P0)
    hover_ekf.setProcessVar(Q)
    hover_ekf.setMeasVar(R)

    # Initialize arrays
    n = len(t)
    accel_true = zeros(n)
    hover_thrust = zeros(n)
    hover_thrust_std = zeros(n)
    hover_thrust_true = zeros(n)
    accel_noise_std = zeros(n)
    innov = zeros(n)
    innov_std = zeros(n)
    innov_test_ratio = zeros(n)
    innov_test_ratio_signed_lpf = zeros(n)
    residual_lpf = zeros(n)

    for k in range(1, n):
        meas_noise_coeff_z = max((abs(vel_z[k]) - vz_thr) + 1.0, 1.0)
        meas_noise_coeff_xy = max((sqrt(vel_x[k]**2 + vel_y[k]**2) - vxy_thr) + 1.0, 1.0)
        hover_ekf.setMeasVarCoeff(max(meas_noise_coeff_xy**2, meas_noise_coeff_z**2))

        # Save data
        hover_thrust[k] = hover_ekf._hover_thr
        hover_thrust_std[k] = sqrt(hover_ekf._P)
        dt = t[k] - t[k-1]

        if dist_bottom[k] > 1.0:
            # Update the EKF
            hover_ekf.predict(dt)
            (innov[k], innov_var, innov_test_ratio[k]) = hover_ekf.fuseAccZ(accel[k], thrust[k])

            innov_std[k] = sqrt(innov_var)
            accel_noise_std[k] = sqrt(hover_ekf._R)
            innov_test_ratio_signed_lpf[k] = hover_ekf._innov_test_ratio_signed_lpf
            residual_lpf[k] = hover_ekf._residual_lpf

    if showplots:
        head_tail = os.path.split(log_name)
        plotData(t, thrust, accel, accel_noise_std, hover_thrust, hover_thrust_std, innov_test_ratio, innov_test_ratio_signed_lpf, innov, innov_std, residual_lpf, head_tail[1])

def plotData(t, thrust, accel, accel_noise_std, hover_thrust, hover_thrust_std, innov_test_ratio, innov_test_ratio_signed_lpf, innov, innov_std, residual_lpf, log_name):
    n_plots = 5
    ax1 = plt.subplot(n_plots, 1, 1)
    ax1.plot(t, thrust * 10.0, '.')
    ax1.plot(t, accel, '.')
    ax1.plot(t, 3*accel_noise_std, 'g--')
    ax1.plot(t, -3*accel_noise_std, 'g--')
    ax1.legend(["Thrust (10x)", "AccZ", "Acc noise est"])
    plt.title(log_name)
    ax1.grid()

    ax2 = plt.subplot(n_plots, 1, 2, sharex=ax1)
    ax2.plot(t, hover_thrust, 'b')
    ax2.plot(t, hover_thrust + hover_thrust_std, 'g--')
    ax2.plot(t, hover_thrust - hover_thrust_std, 'g--')
    ax2.legend(["Ht_Est"])

    ax3 = plt.subplot(n_plots, 1, 3, sharex=ax1)
    ax3.plot(t, hover_thrust_std)
    ax3.legend(["Ht_noise"])

    ax4 = plt.subplot(n_plots, 1, 4, sharex=ax1)
    ax4.plot(t, innov_test_ratio)
    ax4.plot(t, innov_test_ratio_signed_lpf)
    ax4.plot(t, residual_lpf)
    ax4.legend(["Test ratio", "Test ratio lpf", "Residual lpf"])

    ax5 = plt.subplot(n_plots, 1, 5, sharex=ax1)
    ax5.plot(t, innov)
    ax5.plot(t, innov + innov_std, 'g--')
    ax5.plot(t, innov - innov_std, 'g--')
    ax5.legend(["Innov"])
    ax5.grid()

    plt.show()

if __name__ == '__main__':
    import argparse

    # Get the path of this script (without file name)
    script_path = os.path.split(os.path.realpath(__file__))[0]

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Estimate mag biases from ULog file')

    # Provide parameter file path and name
    parser.add_argument('logfile', help='Full ulog file path, name and extension', type=str)
    parser.add_argument('--showplots', help='Display relevant plots',
                        action='store_true')
    args = parser.parse_args()

    logfile = os.path.abspath(args.logfile) # Convert to absolute path

    run(logfile, args.showplots)
