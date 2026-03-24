#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Copyright (c) 2021-2024 PX4 Development Team
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

File: autotune.py
Author: Mathieu Bresciani <mathieu@auterion.com>
License: BSD 3-Clause
Description:
    UI tool for parametric system identification and controller design
"""

import sys

import control as ctrl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from data_extractor import *
from data_selection_window import DataSelectionWindow
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from pid_design import computePidGmvc
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from scipy.signal import detrend
from system_identification import SystemIdentification


def isNumber(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


class Window(QDialog):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        self.model_ref = None
        self.input_ref = None
        self.closed_loop_ref = None
        self.closed_loop_ax = None
        self.bode_plot_ref = []
        self.pz_plot_refs = []
        self.file_name = None
        self.is_system_identified = False
        self.axis = 0
        self.dt = 0.005
        self.rise_time = 0.13
        self.damping_index = 0.0
        self.detune_coeff = 0.5
        self.gains = {"P": 0.01, "I": 0.0, "D": 0.0, "FF": 0.0}
        self.figure = plt.figure(1)
        self.figure.subplots_adjust(hspace=0.5, wspace=1.0)
        self.num = []
        self.den = []
        self.sys_id_delays = 1
        self.sys_id_n_zeros = 2
        self.sys_id_n_poles = 2

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.btn_open_log = QPushButton("Open log")
        self.btn_open_log.clicked.connect(self.loadLog)

        # set the layout
        layout_v = QVBoxLayout()
        layout_h = QHBoxLayout()
        left_menu = QVBoxLayout()
        left_menu.addWidget(self.btn_open_log)

        id_params_group = QFormLayout()
        self.line_edit_zeros = QSpinBox()
        self.line_edit_zeros.setValue(self.sys_id_n_zeros)
        self.line_edit_zeros.setRange(0, 6)
        self.line_edit_zeros.valueChanged.connect(self.onZerosChanged)
        id_params_group.addRow(QLabel("Zeros"), self.line_edit_zeros)
        self.line_edit_poles = QSpinBox()
        self.line_edit_poles.setValue(self.sys_id_n_poles)
        self.line_edit_poles.setRange(0, 6)
        self.line_edit_poles.valueChanged.connect(self.onPolesChanged)
        id_params_group.addRow(QLabel("Poles"), self.line_edit_poles)
        self.line_edit_delays = QSpinBox()
        self.line_edit_delays.setValue(self.sys_id_delays)
        self.line_edit_delays.setRange(0, 1000)
        self.line_edit_delays.valueChanged.connect(self.onDelaysChanged)
        id_params_group.addRow(QLabel("Delays"), self.line_edit_delays)
        input_scale_group = QGroupBox("Input scaling")
        input_scale_group.setToolTip(
            "Scale the input to identify a model at trim airspeed (requires true airspeed data)"
        )

        input_scale_form = QFormLayout()
        self.input_scale_combo = QComboBox()
        self.input_scale_combo.setEditable(False)
        self.input_scale_choices = ["True airspeed^2", "True airspeed", "None"]
        self.input_scale_combo.addItems(self.input_scale_choices)
        self.input_scale_combo.setEnabled(False)
        self.input_scale_combo.currentIndexChanged.connect(self.selectInputScale)
        input_scale_form.addRow(self.input_scale_combo)

        self.line_edit_trim = QDoubleSpinBox()
        self.trim_airspeed = 20.0
        self.line_edit_trim.setValue(self.trim_airspeed)
        self.line_edit_trim.setRange(0.0, 100.0)
        self.line_edit_trim.textChanged.connect(self.onTrimChanged)
        self.line_edit_trim.setEnabled(False)
        input_scale_form.addRow(QLabel("Trim airspeed"), self.line_edit_trim)
        input_scale_group.setLayout(input_scale_form)
        id_params_group.addRow(input_scale_group)

        self.btn_run_sys_id = QPushButton("Run identification")
        self.btn_run_sys_id.clicked.connect(self.onSysIdClicked)
        self.btn_run_sys_id.setEnabled(False)

        id_params_group.addRow(self.btn_run_sys_id)

        left_menu.addLayout(id_params_group)

        layout_tf = self.createTfLayout()
        left_menu.addLayout(layout_tf)

        offset_group = QFormLayout()
        self.line_edit_offset = QDoubleSpinBox()
        self.line_edit_offset.setValue(0.0)
        self.line_edit_offset.setRange(-10.0, 10.0)
        self.line_edit_offset.textChanged.connect(self.onOffsetChanged)
        offset_group.addRow(QLabel("Offset"), self.line_edit_offset)
        left_menu.addLayout(offset_group)
        left_menu.addStretch(1)

        self.tuning_tabs = QTabWidget()

        self.tab_pid = QWidget()
        self.tab_pid.setLayout(self.createPidLayout())
        self.tuning_tabs.addTab(self.tab_pid, "PID")

        self.tab_gmvc = QWidget()
        self.tab_gmvc.setLayout(self.createGmvcLayout())
        self.tuning_tabs.addTab(self.tab_gmvc, "GMVC")

        layout_plot = QVBoxLayout()
        layout_h.addLayout(left_menu)
        layout_h.addLayout(layout_plot)
        layout_h.setStretch(1, 1)
        layout_plot.addWidget(self.toolbar)
        layout_plot.addWidget(self.canvas)
        layout_v.addLayout(layout_h)
        layout_v.setStretch(0, 1)
        layout_v.addWidget(self.tuning_tabs)
        self.setLayout(layout_v)

    def reset(self):
        self.model_ref = None
        self.input_ref = None
        self.closed_loop_ref = None
        self.bode_plot_ref = []
        self.pz_plot_refs = []
        self.is_system_identified = False

    def createTfLayout(self):
        layout_tf = QVBoxLayout()
        self.t_coeffs = QTableWidget()
        self.t_coeffs.setColumnCount(1)
        self.t_coeffs.setHorizontalHeaderLabels(["Coefficients"])
        self.t_coeffs.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.t_coeffs.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.t_coeffs.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.t_coeffs.setFixedWidth(120)
        self.t_coeffs.itemChanged.connect(self.onModelChanged)

        # Place in horizontal layout to center it properly
        self.updateCoeffTable()
        layout_coeff = QHBoxLayout()
        layout_coeff.addWidget(self.t_coeffs)
        layout_tf.addLayout(layout_coeff)

        layout_dt = QFormLayout()
        self.line_edit_dt = QLineEdit("0.0")
        self.line_edit_dt.textChanged.connect(self.onModelChanged)
        layout_dt.addRow(QLabel("dt"), self.line_edit_dt)
        layout_tf.addLayout(layout_dt)

        self.btn_update_model = QPushButton("Update model")
        self.btn_update_model.setEnabled(False)
        self.btn_update_model.clicked.connect(self.updateModel)
        layout_tf.addWidget(self.btn_update_model)
        return layout_tf

    def updateCoeffTable(self):
        self.t_coeffs.setRowCount(self.sys_id_n_poles + self.sys_id_n_zeros + 1)
        self.t_coeffs.clearContents()

        labels = []

        for i in range(self.sys_id_n_poles):
            labels.append("a{}".format(i + 1))

        for i in range(self.sys_id_n_zeros + 1):
            labels.append("b{}".format(i))

        self.t_coeffs.setVerticalHeaderLabels(labels)
        self.t_coeffs.setFixedHeight(
            self.t_coeffs.verticalHeader().length()
            + self.t_coeffs.horizontalHeader().height()
            + 2
        )

    def selectInputScale(self, index):
        self.btn_run_sys_id.setEnabled(True)
        self.plotInputOutput()

    def onModelChanged(self):
        self.btn_update_model.setEnabled(True)

    def onTrimChanged(self):
        try:
            self.trim_airspeed = float(self.line_edit_trim.text())
        except ValueError:
            self.trim_airspeed = 0
            self.line_edit_trim.setValue(self.trim_airspeed)

        self.btn_run_sys_id.setEnabled(True)
        self.plotInputOutput()

    def onOffsetChanged(self):
        self.plotInputOutput()

    def onDelaysChanged(self):
        self.btn_run_sys_id.setEnabled(True)

    def onPolesChanged(self):
        self.sys_id_n_poles = self.line_edit_poles.value()
        self.updateCoeffTable()
        self.btn_run_sys_id.setEnabled(True)

    def onZerosChanged(self):
        self.sys_id_n_zeros = self.line_edit_zeros.value()
        self.updateCoeffTable()
        self.btn_run_sys_id.setEnabled(True)

    def onSysIdClicked(self):
        n_poles = self.line_edit_poles.value()
        n_zeros = self.line_edit_zeros.value()
        self.sys_id_delays = self.line_edit_delays.value()

        if n_poles < n_zeros:
            n_poles = n_zeros
            self.printImproperTfError()

        else:
            self.sys_id_n_zeros = n_zeros
            self.sys_id_n_poles = n_poles
            self.runIdentification()
            self.computeController()

    def printImproperTfError(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Error")
        msg.setText("Transfer function must be proper, set Poles >= Zeros")
        msg.exec_()

    def createPidLayout(self):
        self.gain_line_edit = {}
        self.gain_slider = {}
        self.parallel_gain_lbl = {}
        slider_props = {
            "P": {"min": 0.001, "max": 4.0, "step": 0.001},
            "I": {"min": 0.0, "max": 20.0, "step": 0.1},
            "D": {"min": 0.0, "max": 0.2, "step": 0.001},
            "FF": {"min": -1.0, "max": 1.0, "step": 0.001},
        }

        def make_slider_callback(gain):
            return lambda: self.updateGainFromSlider(gain)

        def make_line_edit_callback(gain):
            return lambda: self.updateGainFromLineEdit(gain)

        layout_pid = QGridLayout()

        layout_options = QHBoxLayout()
        self.pid_no_zero_box = QCheckBox("PI no-zero", self)
        self.pid_no_zero_box.setChecked(False)
        self.pid_no_zero_box.stateChanged.connect(self.updateClosedLoop)
        layout_options.addWidget(self.pid_no_zero_box)

        self.negate_control_box = QCheckBox("Negate control output", self)
        self.negate_control_box.setChecked(False)
        self.negate_control_box.stateChanged.connect(self.updateClosedLoop)
        layout_options.addWidget(self.negate_control_box)
        layout_pid.addLayout(layout_options, 0, 1)

        layout_pid.addWidget(QLabel("Ideal/Standard\nKp * [1 + Ki + Kd]"), 0, 2)
        layout_pid.addWidget(QLabel("Parallel\nKp + Ki + Kd"), 0, 3)

        row = 1
        for gain in self.gains.keys():
            if gain == "FF":
                layout_pid.addWidget(QLabel("{}".format(gain)), row, 0)
            else:
                layout_pid.addWidget(QLabel("K{}".format(gain.lower())), row, 0)

            self.gain_slider[gain] = DoubleSlider(Qt.Horizontal)
            self.gain_slider[gain].setMinimum(slider_props[gain]["min"])
            self.gain_slider[gain].setMaximum(slider_props[gain]["max"])
            self.gain_slider[gain].setInterval(slider_props[gain]["step"])
            self.gain_slider[gain].setValue(self.gains[gain])
            self.gain_slider[gain].valueChanged.connect(make_slider_callback(gain))
            layout_pid.addWidget(self.gain_slider[gain], row, 1)

            self.gain_line_edit[gain] = QLineEdit("{:.3f}".format(self.gains[gain]))
            self.gain_line_edit[gain].setSizePolicy(
                QSizePolicy.Minimum, QSizePolicy.Fixed
            )
            self.gain_line_edit[gain].setMinimumWidth(0)
            self.gain_line_edit[gain].setMinimumSize(0, 0)
            self.gain_line_edit[gain].setAlignment(Qt.AlignCenter)
            self.gain_line_edit[gain].textChanged.connect(make_line_edit_callback(gain))
            layout_pid.addWidget(self.gain_line_edit[gain], row, 2)

            if gain == "P" or gain == "FF":
                self.parallel_gain_lbl[gain] = QLabel("{:.4f}".format(self.gains[gain]))
            else:
                self.parallel_gain_lbl[gain] = QLabel(
                    "{:.4f}".format(self.gains["P"] * self.gains[gain])
                )
            self.parallel_gain_lbl[gain].setAlignment(Qt.AlignCenter)
            layout_pid.addWidget(self.parallel_gain_lbl[gain], row, 3)

            row += 1

        return layout_pid

    def updateGainFromSlider(self, gain: str):
        if self.gain_slider[gain].hasFocus():
            self.gains[gain] = self.gain_slider[gain].value()
            self.gain_line_edit[gain].setText("{:.3f}".format(self.gains[gain]))
            self.updateGainLabels(gain)
            if self.gain_slider[gain].isSliderDown():
                self.updateClosedLoop()

    def updateGainFromLineEdit(self, gain: str):
        if (
            isNumber(self.gain_line_edit[gain].text())
            and self.gain_line_edit[gain].hasFocus()
        ):
            self.gains[gain] = float(self.gain_line_edit[gain].text())
            self.gain_slider[gain].setValue(self.gains[gain])
            self.updateGainLabels(gain)
            self.updateClosedLoop()

    def updateGainLabels(self, gain: str):
        if gain == "FF":
            self.parallel_gain_lbl[gain].setText("{:.4f}".format(self.gains[gain]))
        else:
            # Kp also modifies the Ki and Kd gains of the parallel form
            self.parallel_gain_lbl["P"].setText("{:.4f}".format(self.gains["P"]))
            self.parallel_gain_lbl["I"].setText(
                "{:.4f}".format(self.gains["P"] * self.gains["I"])
            )
            self.parallel_gain_lbl["D"].setText(
                "{:.4f}".format(self.gains["P"] * self.gains["D"])
            )

    def createGmvcLayout(self):
        layout_gmvc = QFormLayout()

        layout_rise_time = QHBoxLayout()
        self.slider_rise_time = DoubleSlider(Qt.Horizontal)
        self.slider_rise_time.setMinimum(0.01)
        self.slider_rise_time.setMaximum(1.0)
        self.slider_rise_time.setInterval(0.01)
        self.slider_rise_time.setValue(self.rise_time)
        self.lbl_rise_time = QLabel("{:.2f}".format(self.rise_time))
        layout_rise_time.addWidget(self.slider_rise_time)
        layout_rise_time.addWidget(self.lbl_rise_time)
        self.slider_rise_time.valueChanged.connect(self.updateLabelRiseTime)
        layout_gmvc.addRow(QLabel("Rise time"), layout_rise_time)

        layout_damping = QHBoxLayout()
        self.slider_damping = DoubleSlider(Qt.Horizontal)
        self.slider_damping.setMinimum(0.0)
        self.slider_damping.setMaximum(2.0)
        self.slider_damping.setInterval(0.1)
        self.slider_damping.setValue(self.damping_index)
        self.lbl_damping = QLabel("{:.1f}".format(self.damping_index))
        layout_damping.addWidget(self.slider_damping)
        layout_damping.addWidget(self.lbl_damping)
        self.slider_damping.valueChanged.connect(self.updateLabelDamping)
        layout_gmvc.addRow(QLabel("Damping index"), layout_damping)

        layout_detune = QHBoxLayout()
        self.slider_detune = DoubleSlider(Qt.Horizontal)
        self.slider_detune.setMinimum(0.0)
        self.slider_detune.setMaximum(2.0)
        self.slider_detune.setInterval(0.1)
        self.slider_detune.setValue(self.detune_coeff)
        self.lbl_detune = QLabel("{:.1f}".format(self.detune_coeff))
        layout_detune.addWidget(self.slider_detune)
        layout_detune.addWidget(self.lbl_detune)
        self.slider_detune.valueChanged.connect(self.updateLabelDetune)
        layout_gmvc.addRow(QLabel("Detune coeff"), layout_detune)
        return layout_gmvc

    def updateLabelRiseTime(self):
        self.rise_time = self.slider_rise_time.value()
        self.lbl_rise_time.setText("{:.2f}".format(self.rise_time))
        if self.slider_rise_time.isSliderDown():
            self.computeController()

    def updateLabelDamping(self):
        self.damping_index = self.slider_damping.value()
        self.lbl_damping.setText("{:.1f}".format(self.damping_index))
        if self.slider_damping.isSliderDown():
            self.computeController()

    def updateLabelDetune(self):
        self.detune_coeff = self.slider_detune.value()
        self.lbl_detune.setText("{:.1f}".format(self.detune_coeff))
        if self.slider_detune.isSliderDown():
            self.computeController()

    def runIdentification(self):
        n_steps = len(self.t)

        n = self.sys_id_n_poles  # order of the denominator (a_1,...,a_n)
        m = self.sys_id_n_zeros  # order of the numerator (b_0,...,b_m)
        d = self.sys_id_delays  # number of delays
        id = SystemIdentification(n, m, d, self.dt)

        est = id.fit(self.u.reshape(-1, 1), self.y.reshape(-1, 1))

        self.num = est.G_.num_list[0][0]
        self.den = est.G_.den_list[0][0][0 : n + 1]
        self.Gz = ctrl.TransferFunction(self.num, self.den, self.dt)

        num_coeffs = self.num
        den_coeffs = self.den[1 : n + 1]
        self.is_system_identified = True
        self.btn_run_sys_id.setEnabled(False)

        self.updateTfDisplay(den_coeffs, num_coeffs)
        self.plotPolesZeros()
        self.replayInputData()

    def replayInputData(self):
        if not self.is_system_identified:
            return
        d = self.sys_id_delays
        u_detrended = detrend(self.u)
        u_delayed = np.concatenate(
            ([0 for k in range(d)], u_detrended[0 : (len(u_detrended) - d)])
        )
        self.t_est, self.y_est = ctrl.forced_response(self.Gz, T=self.t, U=u_delayed)
        if len(self.t_est) > len(self.y_est):
            self.t_est = self.t_est[0 : len(self.y_est - 1)]
        self.plotInputOutput()

    def updateTfDisplay(self, a_coeffs, b_coeffs):

        for i in range(self.sys_id_n_poles):
            self.t_coeffs.setItem(i, 0, QTableWidgetItem("{:.6f}".format(a_coeffs[i])))

        for i in range(self.sys_id_n_zeros + 1):
            self.t_coeffs.setItem(
                self.sys_id_n_poles + i,
                0,
                QTableWidgetItem("{:.6f}".format(b_coeffs[i])),
            )

        dt = self.Gz.dt
        self.line_edit_dt.setText("{:.4f}".format(dt))
        self.btn_update_model.setEnabled(False)

    def plotPolesZeros(self):
        if not self.is_system_identified:
            return
        poles = self.Gz.poles()
        zeros = self.Gz.zeros()
        if not self.pz_plot_refs:
            ax = self.figure.add_subplot(3, 3, 4)
            plot_ref = ax.plot(poles.real, poles.imag, "rx", markersize=10)
            self.pz_plot_refs.append(plot_ref[0])
            plot_ref = ax.plot(zeros.real, zeros.imag, "ro", markersize=10)
            self.pz_plot_refs.append(plot_ref[0])
            uc = mpatches.Circle(
                (0, 0), radius=1, fill=False, color="black", ls="dashed"
            )
            ax.add_patch(uc)
            ax.axhline(0, color="black", linestyle="--")
            ax.axvline(0, color="black", linestyle="--")
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_aspect(1.0)
            ax.set_xlabel("Real")
            ax.set_ylabel("Imag")
            ax.set_title("Pole-Zero Map")
        else:
            self.pz_plot_refs[0].set_xdata(poles.real)
            self.pz_plot_refs[0].set_ydata(poles.imag)
            self.pz_plot_refs[1].set_xdata(zeros.real)
            self.pz_plot_refs[1].set_ydata(zeros.imag)

        self.canvas.draw()

    def updateModel(self):
        self.btn_run_sys_id.setEnabled(True)

        self.den = [1.0]
        self.num = []

        for i in range(self.sys_id_n_poles):
            val = float(self.t_coeffs.item(i, 0).text())
            self.den.append(val)

        for i in range(self.sys_id_n_zeros + 1):
            val = float(self.t_coeffs.item(self.sys_id_n_poles + i, 0).text())
            self.num.append(val)

        dt = float(self.line_edit_dt.text())
        self.dt = dt
        self.Gz = ctrl.TransferFunction(self.num, self.den, self.dt)
        self.resampleData(dt)
        self.is_system_identified = True
        self.plotPolesZeros()
        self.replayInputData()
        self.computeController()
        self.btn_update_model.setEnabled(False)
        return

    def computeController(self):
        if not self.is_system_identified:
            return

        if self.tuning_tabs.tabText(self.tuning_tabs.currentIndex()) == "GMVC":
            sigma = self.rise_time  # rise time
            delta = (
                self.damping_index
            )  # damping property, set between 0 and 2 (1 for Butterworth)
            lbda = self.detune_coeff
            (self.gains["P"], self.gains["I"], self.gains["D"]) = computePidGmvc(
                self.num, self.den, self.dt, sigma, delta, lbda
            )
            # TODO:find a better solution
            self.gains["I"] /= 5.0
            static_gain = sum(self.num) / sum(self.den)
            self.gains["FF"] = max(1 / static_gain, 0.0)

        self.updateKIDSliders()
        self.updateClosedLoop()

    def updateKIDSliders(self):
        for gain in self.gains.keys():
            self.gain_line_edit[gain].setText("{:.3f}".format(self.gains[gain]))
            self.gain_slider[gain].setValue(self.gains[gain])
            self.updateGainLabels(gain)

    def updateClosedLoop(self):
        if not self.is_system_identified:
            return
        # Simulate closed-loop system with generated PID
        num = self.num
        den = self.den
        dt = self.dt
        kc = self.gains["P"]
        ki = self.gains["I"]
        kd = self.gains["D"]
        kff = self.gains["FF"]

        delays = ctrl.TransferFunction(
            [1],
            np.append([1], np.zeros(self.sys_id_delays)),
            dt,
            inputs="r",
            outputs="rd",
        )
        plant = ctrl.TransferFunction(num, den, dt, inputs="u", outputs="plant_out")
        sampler = ctrl.TransferFunction(
            [1], [1, 0], dt, inputs="plant_out", outputs="y"
        )
        sum_feedback = ctrl.summing_junction(inputs=["rd", "-y"], output="e")

        # Default is standard PID
        feedforward = ctrl.TransferFunction(
            [kff], [1], dt, inputs="rd", outputs="ff_out"
        )
        i_control = ctrl.TransferFunction(
            [ki * dt, ki * dt], [2, -2], dt, inputs="e", outputs="i_out"
        )  # Integrator discretized using bilinear transform: s = 2(z-1)/(dt(z+1))

        # Derivative with 1st order LPF (discretized using Euler method: s = (z-1)/dt)
        derivative_cutoff_freq = 10.0  # Hz
        tau = 1 / (2 * np.pi * derivative_cutoff_freq)
        derivative_num = np.array([kd, -kd])
        derivative_den = np.array([tau, -tau + dt])
        d_control = ctrl.TransferFunction(
            derivative_num, derivative_den, dt, inputs="e", outputs="d_out"
        )

        id_control = ctrl.summing_junction(
            inputs=["e", "i_out", "d_out"], output="id_out"
        )
        p_control = ctrl.TransferFunction(
            [kc], [1], dt, inputs="id_out", outputs="pid_out"
        )
        sum_control = ctrl.summing_junction(
            inputs=["pid_out", "ff_out"], output="control_out"
        )

        if self.negate_control_box.isChecked():
            output_sign = -1.0
        else:
            output_sign = 1.0

        out_sign = ctrl.TransferFunction(
            output_sign, 1.0, dt, inputs="control_out", outputs="u"
        )

        remove_zero = self.pid_no_zero_box.isChecked()
        no_derivative_kick = True

        if remove_zero:
            # P on feedback only to remove zero (3-loop autopilot style)
            id_control = ctrl.summing_junction(
                inputs=["-y", "i_out", "d_out"], output="id_out"
            )

        if no_derivative_kick:
            # Derivative on feedback only to remove the "derivative kick"
            d_control = ctrl.TransferFunction(
                -derivative_num, derivative_den, dt, inputs="y", outputs="d_out"
            )

        closed_loop = ctrl.interconnect(
            [
                delays,
                sampler,
                sum_feedback,
                feedforward,
                sum_control,
                p_control,
                i_control,
                d_control,
                id_control,
                out_sign,
                plant,
            ],
            inputs="r",
            outputs="y",
        )

        t_out, y_out = ctrl.step_response(closed_loop, T=np.arange(0, 2, dt))

        # Add disturbance
        sum_feedback_no_ref = ctrl.summing_junction(inputs=["-y"], output="e")
        sum_control_with_disturbance = ctrl.summing_junction(
            inputs=["pid_out", "disturbance"], output="control_out"
        )
        disturbance_loop = ctrl.interconnect(
            [
                sampler,
                sum_feedback_no_ref,
                sum_control_with_disturbance,
                p_control,
                i_control,
                d_control,
                id_control,
                out_sign,
                plant,
            ],
            inputs="disturbance",
            outputs="y",
        )
        d = np.zeros_like(t_out)
        d[t_out >= 1.0] = -0.05  # TODO: parameterize
        _, y_d = ctrl.forced_response(disturbance_loop, t_out, d)
        y_out += y_d

        self.plotClosedLoop(t_out, y_out)

        # Remove feedback
        sum_feedback = ctrl.summing_junction(inputs=["rd"], output="e")

        # Remove feedforward
        sum_control = ctrl.summing_junction(inputs=["pid_out"], output="control_out")

        open_loop = ctrl.interconnect(
            [
                delays,
                sampler,
                sum_feedback,
                sum_control,
                p_control,
                i_control,
                d_control,
                id_control,
                out_sign,
                plant,
            ],
            inputs="r",
            outputs="y",
        )

        self.plotBode(open_loop, closed_loop)

    def plotClosedLoop(self, t, y):
        if self.closed_loop_ref is None:
            ax = self.figure.add_subplot(3, 3, 7)
            ax.step(t, [1 if i > 0 else 0 for i in t], "k--")
            plot_ref = ax.plot(t, y)
            self.closed_loop_ref = plot_ref[0]
            self.closed_loop_ax = ax
            ax.set_title("Closed-loop step response")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude (rad/s)")
        else:
            self.closed_loop_ref.set_xdata(t)
            self.closed_loop_ref.set_ydata(y)
            self.closed_loop_ax.set_ylim(np.min(y), np.max([1.5, np.max(y)]))

        self.canvas.draw()

    def plotBode(self, open_loop, closed_loop):

        (
            gain_margin,
            phase_margin,
            stab_margin,
            phase_crossover,
            gain_crossover,
            stab_margin_w,
        ) = ctrl.stability_margins(open_loop)
        stability_margins_text = f"Gain margin: {20 * np.log10(gain_margin):.2f}dB (@{phase_crossover / (2 * np.pi):.1f}Hz)\nPhase margin: {phase_margin:.1f}deg (@{gain_crossover / (2 * np.pi):.1f}Hz)"

        w = np.logspace(-1, 3, 40).tolist()
        (mag_ol, phase_ol, omega_ol) = ctrl.frequency_response(
            open_loop, omega=np.asarray(w)
        )

        (mag_cl, phase_cl, omega_cl) = ctrl.frequency_response(
            closed_loop, omega=np.asarray(w)
        )
        f = omega_cl / (2 * np.pi)

        if not self.bode_plot_ref:
            ax = self.figure.add_subplot(3, 3, (5, 6))
            plot_ref = ax.semilogx(f, 20 * np.log10(mag_ol), label="Open-loop")
            self.bode_plot_ref.append(plot_ref[0])
            plot_ref = ax.semilogx(f, 20 * np.log10(mag_cl), label="Closed-loop")
            self.bode_plot_ref.append(plot_ref[0])
            ax.set_ylim(-20, 20)
            ax.plot([f[0], f[-1]], [0, 0], "k--")
            ax.plot([f[0], f[-1]], [-3, -3], "g--")

            ax.set_title("Bode")
            ax.set_ylabel("Magnitude (dB)")
            ax.legend()

            ax = self.figure.add_subplot(3, 3, (8, 9))
            plot_ref = ax.semilogx(f, phase_ol * 180 / np.pi, label="Open-loop")
            self.bode_plot_ref.append(plot_ref[0])
            plot_ref = ax.semilogx(f, phase_cl * 180 / np.pi, label="Closed-loop")
            self.bode_plot_ref.append(plot_ref[0])
            ax.set_ylim(-180, 180)

            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Phase (deg)")
            self.gain_margin_text_ref = ax.text(
                0.01,
                0.9,
                stability_margins_text,
                verticalalignment="top",
                transform=ax.transAxes,
            )

        else:
            self.bode_plot_ref[0].set_xdata(f)
            mag_ol_db = 20 * np.log10(mag_ol)
            self.bode_plot_ref[0].set_ydata(mag_ol_db)
            self.bode_plot_ref[1].set_xdata(f)
            mag_cl_db = 20 * np.log10(mag_cl)
            self.bode_plot_ref[1].set_ydata(mag_cl_db)
            self.gain_margin_text_ref.set_text(stability_margins_text)

            self.bode_plot_ref[2].set_xdata(f)
            self.bode_plot_ref[2].set_ydata(phase_ol * 180 / np.pi)
            self.bode_plot_ref[3].set_xdata(f)
            self.bode_plot_ref[3].set_ydata(phase_cl * 180 / np.pi)

        self.canvas.draw()

    def plotInputOutput(self, redraw=False):
        if len(self.true_airspeed) == len(self.input):
            scale = 1

            scale_type = self.input_scale_choices[self.input_scale_combo.currentIndex()]
            if scale_type == "True airspeed":
                scale = np.array(self.true_airspeed) / self.trim_airspeed

            elif scale_type == "True airspeed^2":
                scale = (np.array(self.true_airspeed) / self.trim_airspeed) ** 2

            self.u = self.input * scale
            self.input_scale_combo.setEnabled(True)
            self.line_edit_trim.setEnabled(True)
        else:
            self.input_scale_combo.setEnabled(False)
            self.line_edit_trim.setEnabled(False)

        if self.model_ref is None or redraw:
            # First time we have no plot reference, so do a normal plot.
            # .plot returns a list of line <reference>s, as we're
            # only getting one we can take the first element.
            self.figure.clear()
            ax = self.figure.add_subplot(3, 3, (1, 3))
            input_ref = ax.plot(self.t, self.u)
            self.input_ref = input_ref[0]
            ax.plot(self.t, self.y)
            plot_refs = ax.plot(0, 0)
            self.model_ref = plot_refs[0]
            ax.set_title("Logged data")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.legend(["Input", "Output", "Model"])
        else:
            # We have a reference, we can use it to update the data for that line.
            self.model_ref.set_xdata(self.t_est)
            try:
                offset = float(self.line_edit_offset.text())
            except ValueError:
                offset = 0

            self.model_ref.set_ydata(self.y_est + offset)
            self.input_ref.set_ydata(self.u)

        self.canvas.draw()

    def loadLog(self):
        select = DataSelectionWindow(self.file_name)

        if select.exec_():
            self.reset()
            self.file_name = select.file_name
            self.t = select.t - select.t[0]
            self.input = select.u
            self.u = self.input
            self.y = select.y
            self.true_airspeed = select.v
            trim_airspeed = select.getTrimAirspeed()

            if trim_airspeed is not None:
                self.line_edit_trim.setValue(trim_airspeed)

            self.refreshInputOutputData()
            self.runIdentification()
            self.computeController()

    def refreshInputOutputData(self):
        self.reset()
        if self.file_name:
            dt = max(get_delta_mean(self.t), 0.008)
            self.resampleData(dt)
            self.plotInputOutput(redraw=True)

    def resampleData(self, dt):
        self.dt = dt
        t_new = np.arange(0, self.t[-1] + self.dt, self.dt)
        self.u = resample_interp(self.t, self.u, t_new)
        self.y = resample_interp(self.t, self.y, t_new)
        self.input = resample_interp(self.t, self.input, t_new)

        if len(self.true_airspeed) > 0:
            self.true_airspeed = resample_interp(self.t, self.true_airspeed, t_new)

        self.t = t_new


class DoubleSlider(QSlider):

    def __init__(self, *args, **kargs):
        super(DoubleSlider, self).__init__(*args, **kargs)
        self._min = 0
        self._max = 99
        self.interval = 1

    def setValue(self, value):
        index = round((value - self._min) / self.interval)
        return super(DoubleSlider, self).setValue(int(index))

    def value(self):
        return self.index * self.interval + self._min

    @property
    def index(self):
        return super(DoubleSlider, self).value()

    def setIndex(self, index):
        return super(DoubleSlider, self).setValue(index)

    def setMinimum(self, value):
        self._min = value
        self._range_adjusted()

    def setMaximum(self, value):
        self._max = value
        self._range_adjusted()

    def setInterval(self, value):
        # To avoid division by zero
        if not value:
            raise ValueError("Interval of zero specified")
        self.interval = value
        self._range_adjusted()

    def _range_adjusted(self):
        number_of_steps = int((self._max - self._min) / self.interval)
        super(DoubleSlider, self).setMaximum(number_of_steps)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())
