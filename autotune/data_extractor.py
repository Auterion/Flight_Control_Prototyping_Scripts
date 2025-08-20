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


class FieldDefinition:
    def __init__(self, topic, var, inst):
        self.topic_name = topic
        self.variable_name = var
        self.instance = inst


class DataExtractor:
    def __init__(self, logfile_name):
        self.log = ULog(logfile_name)

    def get_topics_list(self):
        fields = []
        for elem in self.log.data_list:
            for var_name in elem.data.keys():
                fields.append(FieldDefinition(elem.name, var_name, elem.multi_id))

        return fields

    def getPreview(self, field_def):
        (t_data, data) = self.getData(field_def)

        if len(t_data) > 10e3:
            # Downsample to speed up plotting preview
            downsampling_factor = int(len(t_data) / 10e3) + 1
            t_data = t_data[: -downsampling_factor + 1 : downsampling_factor]
            data = data[: -downsampling_factor + 1 : downsampling_factor]

        return (t_data, data)

    def getTrimAirspeed(self):
        params = self.log.initial_parameters
        if "FW_AIRSPD_TRIM" in params:
            return params["FW_AIRSPD_TRIM"]
        else:
            return None

    def getData(self, field_def):
        data = get_data(
            self.log, field_def.topic_name, field_def.variable_name, field_def.instance
        )
        t_data = us2s(
            get_data(self.log, field_def.topic_name, "timestamp", field_def.instance)
        )

        return (t_data, data)

    def getInputOutputData(self, field_def_u, field_def_y, t_start=0.0, t_stop=0.0):
        (t_u_data, u_data) = self.getData(field_def_u)
        (t_y_data, y_data) = self.getData(field_def_y)

        v_data = get_data(self.log, "airspeed_validated", "true_airspeed_m_s")
        t_v_data = us2s(get_data(self.log, "airspeed_validated", "timestamp"))

        (t_aligned, u_aligned, y_aligned, v_aligned) = resampleIdentificationData(
            t_u_data, u_data, t_y_data, y_data, t_v_data, v_data, t_start, t_stop
        )

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
    for i in range(1, length):
        dx = dx + (data_list[i] - data_list[i - 1])

    dx = dx / (length - 1)
    return dx


def resample_interp(t, u, t_new):
    t_unique, indices = np.unique(t, return_index=True)
    interp = make_interp_spline(t_unique, u[indices], k=1)
    return interp(t_new)


def find_autotune_sequence(log, axis):
    t_start = None
    t_stop = None
    status_data = get_data(log, "autotune_attitude_control_status", "state")
    t_status = us2s(get_data(log, "autotune_attitude_control_status", "timestamp"))
    axis_to_state = [2, 4, 6]  # roll, pitch, yaw states

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

    return (t_start, t_stop)


def resampleIdentificationData(
    t_u_data, u_data, t_y_data, y_data, t_v_data, v_data, t_start, t_stop
):
    if not t_start:
        t_start = t_u_data[0]

    if not t_stop:
        t_stop = t_u_data[-1]

    dt = get_delta_mean(t_y_data)
    t_aligned = np.arange(t_start, t_stop, dt)

    # Resample series to the common index
    u_aligned = resample_interp(t_u_data, u_data, t_aligned)
    y_aligned = resample_interp(t_y_data, y_data, t_aligned)

    v_aligned = []

    if len(v_data) > 0:
        v_data = np.nan_to_num(v_data, nan=1000)
        v_aligned = resample_interp(t_v_data, v_data, t_aligned)

    return (t_aligned, u_aligned, y_aligned, v_aligned)


def printCppArrays(t_aligned, u_aligned, y_aligned):
    # Print data in c++ arrays
    # TODO: print to file and trigger from GUI using an "export" button
    n_samples = len(t_aligned)
    u_array = "static constexpr float u_data[{}] = {{".format(n_samples)
    y_array = "static constexpr float y_data[{}] = {{".format(n_samples)
    t_array = "static constexpr float t_data[{}] = {{".format(n_samples)

    for u in u_aligned:
        u_array += "{}f, ".format(u)

    for y in y_aligned:
        y_array += "{}f, ".format(y)

    for t in t_aligned:
        t_array += "{}f, ".format(t)

    u_array += "};"
    y_array += "};"
    t_array += "};"

    print("\n")
    print(u_array)
    print("\n")
    print(y_array)
    print("\n")
    print(t_array)


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Extract identification data from a give .ulg file"
    )

    parser.add_argument(
        "logfile", help="Full ulog file path, name and extension", type=str
    )
    args = parser.parse_args()

    logfile = os.path.abspath(args.logfile)  # Convert to absolute path

    x_field_def = FieldDefinition("vehicle_torque_setpoint", "xyz[0]", 0)
    y_field_def = FieldDefinition("vehicle_angular_velocity", "xyz[0]", 0)
    (t_aligned, u_aligned, y_aligned, v_aligned) = DataExtractor(
        logfile
    ).getInputOutputData(x_field_def, y_field_def)
    printCppArrays(t_aligned, u_aligned, y_aligned)
