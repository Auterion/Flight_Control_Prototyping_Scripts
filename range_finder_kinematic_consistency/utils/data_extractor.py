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

File: data_extractor.py
Author: Mathieu Bresciani <mathieu@auterion.com>
License: BSD 3-Clause
Description:
"""

import numpy as np
from scipy import signal
from pyulog import ULog

def getAllData(logfile):
    log = ULog(logfile)

    rng = getData(log, 'distance_sensor', 'current_distance')
    t_rng = ms2s(getData(log, 'distance_sensor', 'timestamp'))

    vz = getData(log, 'vehicle_local_position', 'vz')
    t_vz = ms2s(getData(log, 'vehicle_local_position', 'timestamp'))

    STATE_VZ = 6
    vz_var = getData(log, 'estimator_states', f'covariances[{STATE_VZ}]')
    t_vz_var = ms2s(getData(log, 'estimator_states', 'timestamp'))

    (t_aligned, rng_aligned, vz_aligned, vz_var_aligned) = alignData(log, t_rng, rng, t_vz, vz, t_vz_var, vz_var)

    t_aligned -= t_aligned[0]

    return (t_aligned, rng_aligned, vz_aligned, vz_var_aligned)

def getData(log, topic_name, variable_name, instance=0):
    variable_data = np.array([])
    for elem in log.data_list:
        if elem.name == topic_name:
            if instance == elem.multi_id:
                variable_data = elem.data[variable_name]
                break

    return variable_data

def ms2s(time_ms):
    return time_ms * 1e-6

def getDeltaMean(data_list):
    dx = 0
    length = len(data_list)
    for i in range(1,length):
        dx = dx + (data_list[i]-data_list[i-1])

    dx = dx/(length-1)
    return dx

def alignData(log, t_u_data, u_data, t_y_data, y_data, t_y2_data, y2_data):
    len_y = len(t_y_data)
    len_y2 = len(t_y2_data)
    i_y = 0
    i_y2 = 0
    u_aligned = []
    y_aligned = []
    y2_aligned = []
    t_aligned = []

    for i_u in range(len(t_u_data)):
        t_u = t_u_data[i_u]

        while t_y_data[i_y] <= t_u and i_y < len_y-1:
            i_y += 1
        while t_y2_data[i_y2] <= t_u and i_y2 < len_y2-1:
            i_y2 += 1

        u_aligned = np.append(u_aligned, u_data[i_u])
        y_aligned = np.append(y_aligned, y_data[i_y-1])
        y2_aligned = np.append(y2_aligned, y2_data[i_y2-1])
        t_aligned.append(t_u)

    return (t_aligned, u_aligned, y_aligned, y2_aligned)

if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description='Extract data from a give .ulg file')

    parser.add_argument('logfile', help='Full ulog file path, name and extension', type=str)
    args = parser.parse_args()

    logfile = os.path.abspath(args.logfile) # Convert to absolute path

    (t_aligned, u_aligned, y_aligned, y2_data) = getAllData(logfile)
