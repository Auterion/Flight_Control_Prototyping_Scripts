#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Copyright (c) 2021 PX4 Development Team
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

File: data_extractor.py
Author: Mathieu Bresciani <mathieu@auterion.com>
License: BSD 3-Clause
Description:
    rate controller auto-tuning algorithm test on real data
"""

import numpy as np
from scipy import signal
from pyulog import ULog
from scipy.interpolate import make_interp_spline

def getInputOutputData(logfile, axis, t_start=0.0, t_stop=0.0, instance=0):
    log = ULog(logfile)

    y_data = get_data(log, 'vehicle_angular_velocity', 'xyz[{}]'.format(axis))
    t_y_data = us2s(get_data(log, 'vehicle_angular_velocity', 'timestamp'))

    u_data = get_data(log, 'vehicle_torque_setpoint', 'xyz[{}]'.format(axis))
    t_u_data = us2s(get_data(log, 'vehicle_torque_setpoint', 'timestamp'))

    v_data = get_data(log, 'airspeed_validated', 'true_airspeed_m_s')
    t_v_data = us2s(get_data(log, 'airspeed_validated', 'timestamp'))

    if not np.any(u_data):
        # Check for legacy topics
        actuator_controls_n = 'actuator_controls_{}'.format(instance)
        u_data = get_data(log, actuator_controls_n, 'control[{}]'.format(axis))
        t_u_data = us2s(get_data(log, actuator_controls_n, 'timestamp'))

    (t_aligned, u_aligned, y_aligned, v_aligned) = extract_identification_data(log, t_u_data, u_data, t_y_data, y_data, axis, t_v_data, v_data, t_start, t_stop)

    return (t_aligned, u_aligned, y_aligned, v_aligned)

def get_data(log, topic_name, variable_name, instance=0):
    variable_data = np.array([])
    for elem in log.data_list:
        if elem.name == topic_name:
            if instance == elem.multi_id:
                variable_data = elem.data[variable_name]
                break

    return variable_data

def us2s(time_ms):
    return time_ms * 1e-6

def get_delta_mean(data_list):
    dx = 0
    length = len(data_list)
    for i in range(1,length):
        dx = dx + (data_list[i]-data_list[i-1])

    dx = dx/(length-1)
    return dx

def resample_interp(t, u, t_new):
    interp = make_interp_spline(t, u, k=1)
    return interp(t_new)

def extract_identification_data(log, t_u_data, u_data, t_y_data, y_data, axis, t_v_data, v_data, t_start, t_stop):
    if t_start == 0.0 and t_stop == 0.0:
        # Find autotune sequence
        status_data = get_data(log, 'autotune_attitude_control_status', 'state')
        t_status = us2s(get_data(log, 'autotune_attitude_control_status', 'timestamp'))
        axis_to_state = [2, 4, 6] # roll, pitch, yaw states

        status_prev = 0

        for i_s in range(len(t_status)):
            if status_data[i_s] == axis_to_state[axis]:
                if status_prev != axis_to_state[axis]:
                    t_start = t_status[i_s]

            else:
                if status_prev == axis_to_state[axis]:
                    t_stop = t_status[i_s]
                    break

            status_prev = status_data[i_s]

    if t_start == 0.0:
        t_start = t_u_data[0]

    if t_stop == 0.0:
        t_stop = t_u_data[-1]

    dt = get_delta_mean(t_y_data)
    t_aligned = np.arange(t_start, t_stop, dt)

    # Resample series to the common index
    u_aligned = resample_interp(t_u_data, u_data, t_aligned)
    y_aligned = resample_interp(t_y_data, y_data, t_aligned)

    v_aligned = []

    if len(v_data) > 0:
        v_aligned = resample_interp(t_v_data, v_data, t_aligned)

    return (t_aligned, u_aligned, y_aligned, v_aligned)

def printCppArrays(t_aligned, u_aligned, y_aligned):
    # Print data in c++ arrays
    # TODO: print to file and trigger from GUI using an "export" button
    n_samples = len(t_aligned)
    u_array = 'static constexpr float u_data[{}] = {{'.format(n_samples)
    y_array = 'static constexpr float y_data[{}] = {{'.format(n_samples)
    t_array = 'static constexpr float t_data[{}] = {{'.format(n_samples)

    for u in u_aligned:
        u_array += '{}f, '.format(u)

    for y in y_aligned:
        y_array += '{}f, '.format(y)

    for t in t_aligned:
        t_array += '{}f, '.format(t)

    u_array += '};'
    y_array += '};'
    t_array += '};'

    print('\n')
    print(u_array)
    print('\n')
    print(y_array)
    print('\n')
    print(t_array)

if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description='Extract identification data from a give .ulg file')

    parser.add_argument('logfile', help='Full ulog file path, name and extension', type=str)
    parser.add_argument('--axis', dest='axis', choices=['x', 'y', 'z'], help='the body axis on interest')
    args = parser.parse_args()

    logfile = os.path.abspath(args.logfile) # Convert to absolute path
    axis = {'x':0, 'y':1, 'z':2}[args.axis]

    (t_aligned, u_aligned, y_aligned, v_aligned) = getInputOutputData(logfile, axis, instance=0)
    printCppArrays(t_aligned, u_aligned, y_aligned)
