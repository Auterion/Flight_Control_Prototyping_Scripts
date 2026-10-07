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
from loop_model import LoopModel, idealGains, parallelGains
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
from pid_design import computePidGmvc
from PyQt5.QtCore import Qt, QThread, pyqtSignal
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
    QMenu,
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
    QWidgetAction,
)
from scipy.signal import detrend
from system_identification import SystemIdentification, arx_transfer_function


def computeNRMSE(y, y_est):
    # Normalized Root Mean Square Error (NRMSE) expressed as a percentage.
    norm_ref = np.linalg.norm(y - np.mean(y))
    if norm_ref < 1e-10:
        return -np.inf
    return 100.0 * (1.0 - np.linalg.norm(y - y_est) / norm_ref)


def replayModel(Gz, delay, t, u):
    """Output of the model driven by the detrended logged input."""
    u_detrended = detrend(u)
    u_delayed = np.concatenate(([0] * delay, u_detrended[: len(u_detrended) - delay]))
    with np.errstate(over="ignore", invalid="ignore"):
        _, y_est = ctrl.forced_response(Gz, T=t, U=u_delayed)
    return y_est


def replayFit(y, y_est):
    if not np.all(np.isfinite(y_est)):
        # Replay of an unstable model
        return -np.inf
    return computeNRMSE(detrend(y[: len(y_est)]), detrend(y_est))


def compute_fit(u, y, t, dt, n_poles, n_zeros, delay, f_hp, f_lp, method="RLS"):
    try:
        sys_id = SystemIdentification(n_poles, n_zeros, delay, dt)
        sys_id.f_hp = f_hp
        sys_id.f_lp = f_lp
        est = sys_id.fit(u, y, method=method)
        return replayFit(y, replayModel(est.G_, delay, t, u))
    except Exception:
        return -np.inf


kGainFormLabels = {
    "ideal": "Ideal/Standard\nKp * [1 + Ki + Kd]",
    "parallel": "Parallel\nKp + Ki + Kd",
}
kGainFormat = {"ideal": "{:.3f}", "parallel": "{:.4f}"}
# Loaded when the log gives no flown gains: the lowest slider values, except
# FF left at 0 (no feedforward) rather than its -1 slider end
kMinimumGains = {"P": 0.001, "I": 0.0, "D": 0.0, "FF": 0.0}
# (min, max, step) of each slider. Parallel Ki and Kd are the ideal ones
# scaled by Kp, typically below 1: finer steps keep them from rounding to 0
kGainSliderRanges = {
    "ideal": {
        "P": (0.001, 4.0, 0.001),
        "I": (0.0, 20.0, 0.1),
        "D": (0.0, 0.2, 0.001),
        "FF": (-1.0, 1.0, 0.001),
    },
    "parallel": {
        "P": (0.001, 4.0, 0.001),
        "I": (0.0, 20.0, 0.01),
        "D": (0.0, 0.05, 0.0001),
        "FF": (-1.0, 1.0, 0.001),
    },
}


def compute_closed_loop_fit(num, den, dt, delays, gains, t, r, y, **pid_options):
    """Fit of the logged output replayed from the logged reference in closed loop.

    pid_options are the LoopModel options of the controller. Returns None when
    the replayed loop is unstable.
    """
    loop = LoopModel(num, den, dt, delays, gains, **pid_options)
    if not loop.isStable():
        return None
    y_replay = loop.simulate(t, r, np.zeros_like(t))
    return computeNRMSE(detrend(y), detrend(y_replay))


def showFit(label, fit):
    label.setText(f"{fit:.1f}%")
    if fit >= 80:
        label.setStyleSheet("color: green")
    elif fit >= 60:
        label.setStyleSheet("color: orange")
    else:
        label.setStyleSheet("color: red")


class ParamSearchWorker(QThread):
    finished = pyqtSignal(dict, float)
    progress = pyqtSignal(int, int)

    def __init__(self, u, y, t, dt, f_hp, f_lp, method):
        super().__init__()
        self._cancel = False
        self.u = u
        self.y = y
        self.t = t
        self.dt = dt
        self.f_hp = f_hp
        self.f_lp = f_lp
        self.method = method

    def run(self):
        n_poles_range = range(2, 8)
        n_zeros_range = range(2, 8)
        delay_range = range(0, 6)

        combos = [
            (n_poles, n_zeros, delay)
            for n_poles in n_poles_range
            for n_zeros in n_zeros_range
            for delay in delay_range
            if n_poles >= n_zeros
        ]
        total = len(combos)
        best_fit = -np.inf
        best_params = {}

        for i, (n_poles, n_zeros, delay) in enumerate(combos):
            fit = compute_fit(
                self.u,
                self.y,
                self.t,
                self.dt,
                n_poles,
                n_zeros,
                delay,
                self.f_hp,
                self.f_lp,
                self.method,
            )
            new_order = n_poles
            best_order = best_params.get("n_poles", 0)
            # Prefer lower-order models: a higher-order model must improve fit
            # by more than 1% to be accepted, avoiding overfitting.
            if (fit > best_fit + 1.0) or (fit > best_fit and new_order == best_order):
                best_fit = fit
                best_params = {
                    "n_poles": n_poles,
                    "n_zeros": n_zeros,
                    "delay": delay,
                }
            self.progress.emit(i + 1, total)
            if self._cancel:
                break

        self.finished.emit(best_params, best_fit)


def thresholdColor(value, limit, limit_is_minimum):
    """Green when `value` respects `limit`, red when it violates it, None if unknown."""
    if value is None or not np.isfinite(value):
        return None
    respected = value >= limit if limit_is_minimum else value <= limit
    return "green" if respected else "red"


def excludeAnnotationsFromLayout(ax):
    # Texts and legends are drawn inside the axes: when constrained layout
    # makes room for them, small axes get larger margins, shrink further and
    # collapse
    for artist in [*ax.texts, *ax.artists, ax.get_legend()]:
        if artist is not None:
            artist.set_in_layout(False)


def drawZPlaneGrid(ax, f_nyquist):
    """Lines of constant damping ratio and natural frequency in the z-plane."""
    style = dict(color="gray", linewidth=0.4, alpha=0.7)
    unit_circle = np.exp(1j * np.linspace(0, 2 * np.pi, 200))
    ax.plot(unit_circle.real, unit_circle.imag, "k:", linewidth=0.8)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5)

    # z = exp(s * dt) with s = wn * (-zeta + j * sqrt(1 - zeta^2)), drawn
    # up to the Nyquist frequency where the damped angle wd * dt reaches pi
    damped_angle = np.linspace(0, np.pi, 100)
    for zeta in np.arange(0.1, 1.0, 0.1):
        z = np.exp(-zeta / np.sqrt(1 - zeta**2) * damped_angle) * np.exp(
            1j * damped_angle
        )
        ax.plot(z.real, z.imag, **style)
        ax.plot(z.real, -z.imag, **style)
        label_point = z[len(z) // 2]
        ax.text(
            label_point.real, label_point.imag, f"{zeta:.1f}", fontsize=6, color="gray"
        )

    zeta = np.linspace(0, 1, 100)
    for fraction in np.arange(0.1, 1.01, 0.1):
        z = np.exp(fraction * np.pi * (-zeta + 1j * np.sqrt(1 - zeta**2)))
        ax.plot(z.real, z.imag, **style)
        ax.plot(z.real, -z.imag, **style)
        ax.text(
            z[0].real,
            z[0].imag,
            f"{fraction * f_nyquist:.0f}Hz",
            fontsize=6,
            color="gray",
        )


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
        self.closed_loop_step_ref = None
        self.closed_loop_ax = None
        self.measured_step_info = None
        self.step_info_patches = []
        self.step_info_spinbox = {}
        self.step_info_measured_lbl = {}
        self.step_info = {
            "rise_time": 0.12,
            "overshoot": 10.0,
            "settling_time": 0.4,
        }
        self.bode_plot_ref = []
        self.margin_text_refs = {}
        self.pz_plot_refs = []
        self.file_name = None
        # Windows (s since boot) the model is validated on, and their
        # (t, u, y, v) data
        self.validation_windows = []
        self.validation_data = []
        self.is_system_identified = False
        self.axis = 0
        self.dt = 0.005
        self.rise_time = 0.13
        self.damping_index = 0.0
        self.detune_coeff = 0.5
        self.gains = {"P": 0.01, "I": 0.0, "D": 0.0, "FF": 0.0}  # ideal form
        self.gain_form = "ideal"
        self.reference = None
        self.flown_gains = None
        self.flown_gain_form = None
        self.flown_options = {}
        self.figure = plt.figure(1, layout="constrained")
        self.num = []
        self.den = []
        self.sys_id_delays = 1
        self.sys_id_n_zeros = 2
        self.sys_id_n_poles = 2

        self.kDisturbanceTime = 1.0
        self.kMinGainMarginDb = 6.0
        self.kMinPhaseMarginDeg = 45.0
        # Equivalent to a peak sensitivity |S|max <= 2 (6dB)
        self.kMinModulusMargin = 0.5
        # The identified delay is an integer number of samples: the loop should
        # survive it being underestimated by one
        self.kMinDelayMarginSamples = 1.0
        self.kUnstableLoopText = "Unstable closed loop: margins do not apply"
        self.step_duration = 2.0
        self.disturbance_amplitude = -0.05
        self.step_sim_spinbox = {}

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.robustness_figure = plt.figure(2, layout="constrained")
        self.robustness_canvas = FigureCanvas(self.robustness_figure)
        self.robustness_toolbar = NavigationToolbar(self.robustness_canvas, self)
        self.nyquist_ax = None
        self.sensitivity_ax = None
        self.root_locus_ax = None

        self.btn_open_log = QPushButton("Open log")
        self.btn_open_log.clicked.connect(self.loadLog)

        # set the layout
        layout_v = QVBoxLayout()
        layout_h = QHBoxLayout()
        left_menu = QVBoxLayout()
        left_menu.addWidget(self.btn_open_log)

        id_params_group = QFormLayout()
        self.createPreprocessingWidget(id_params_group)
        self.createFindParamsButton(id_params_group)
        self.createModelOrderWidgets(id_params_group)
        self.createMethodWidget(id_params_group)
        self.createRunSysIdButton(id_params_group)
        self.createFitWidget(id_params_group)
        self.createStabilityWidget(id_params_group)

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

        self.plot_tabs = QTabWidget()
        self.plot_tabs.addTab(self.createPlotTab(self.toolbar, self.canvas), "General")
        self.plot_tabs.addTab(
            self.createPlotTab(self.robustness_toolbar, self.robustness_canvas),
            "Robustness",
        )
        self.validation_figure = Figure(layout="constrained")
        self.validation_canvas = FigureCanvas(self.validation_figure)
        self.validation_tab = self.createPlotTab(
            NavigationToolbar(self.validation_canvas, self), self.validation_canvas
        )
        self.plot_tabs.addTab(self.validation_tab, "Validation")
        self.plot_tabs.currentChanged.connect(self.onPlotTabChanged)

        layout_h.addLayout(left_menu)
        layout_h.addWidget(self.plot_tabs)
        layout_h.setStretch(1, 1)
        layout_v.addLayout(layout_h)
        layout_v.setStretch(0, 1)
        bottom_row = QHBoxLayout()
        bottom_row.addWidget(self.tuning_tabs, stretch=1)
        bottom_row.addWidget(self.createStepInfoGroup())
        layout_v.addLayout(bottom_row)
        self.setLayout(layout_v)

    def reset(self):
        self.model_ref = None
        self.input_ref = None
        self.closed_loop_ref = None
        self.closed_loop_step_ref = None
        self.measured_step_info = None
        self.step_info_patches = []
        self.lbl_fit.setText("—")
        self.lbl_fit.setStyleSheet("")
        self.lbl_closed_loop_fit.setText("—")
        self.lbl_closed_loop_fit.setStyleSheet("")
        self.lbl_stability.setText("—")
        self.lbl_stability.setStyleSheet("")
        self.btn_stabilize.setVisible(False)
        self.bode_plot_ref = []
        self.margin_text_refs = {}
        self.pz_plot_refs = []
        self.is_system_identified = False
        self.robustness_figure.clear()
        self.nyquist_ax = None
        self.sensitivity_ax = None
        self.root_locus_ax = None
        self.robustness_canvas.draw()

    def createPlotTab(self, toolbar, canvas):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        tab.setLayout(layout)
        return tab

    def createModelOrderWidgets(self, layout):
        self.line_edit_zeros = QSpinBox()
        self.line_edit_zeros.setValue(self.sys_id_n_zeros)
        self.line_edit_zeros.setRange(0, 6)
        self.line_edit_zeros.valueChanged.connect(self.onZerosChanged)
        layout.addRow(QLabel("Zeros"), self.line_edit_zeros)
        self.line_edit_poles = QSpinBox()
        self.line_edit_poles.setValue(self.sys_id_n_poles)
        self.line_edit_poles.setRange(0, 6)
        self.line_edit_poles.valueChanged.connect(self.onPolesChanged)
        layout.addRow(QLabel("Poles"), self.line_edit_poles)
        self.line_edit_delays = QSpinBox()
        self.line_edit_delays.setValue(self.sys_id_delays)
        self.line_edit_delays.setRange(0, 1000)
        self.line_edit_delays.valueChanged.connect(self.onDelaysChanged)
        layout.addRow(QLabel("Delays"), self.line_edit_delays)

    def createMethodWidget(self, layout):
        self.id_method_combo = QComboBox()
        self.id_method_combo.addItems(["OLS", "RLS"])
        self.id_method_combo.currentIndexChanged.connect(
            lambda: self.btn_run_sys_id.setEnabled(True)
        )
        layout.addRow(QLabel("Method"), self.id_method_combo)

    def createPreprocessingWidget(self, layout):
        preproc_group = QGroupBox("Pre-processing")
        preproc_form = QFormLayout()
        self.f_hp_spinbox = QDoubleSpinBox()
        self.f_hp_spinbox.setRange(0.0, 50.0)
        self.f_hp_spinbox.setSingleStep(0.1)
        self.f_hp_spinbox.setDecimals(1)
        self.f_hp_spinbox.setValue(0.0)
        self.f_hp_spinbox.valueChanged.connect(
            lambda: self.btn_run_sys_id.setEnabled(True)
        )
        preproc_form.addRow(QLabel("HP cutoff (Hz)"), self.f_hp_spinbox)
        self.f_lp_spinbox = QDoubleSpinBox()
        self.f_lp_spinbox.setRange(1.0, 200.0)
        self.f_lp_spinbox.setSingleStep(1.0)
        self.f_lp_spinbox.setDecimals(1)
        self.f_lp_spinbox.setValue(30.0)
        self.f_lp_spinbox.valueChanged.connect(
            lambda: self.btn_run_sys_id.setEnabled(True)
        )
        preproc_form.addRow(QLabel("LP cutoff (Hz)"), self.f_lp_spinbox)
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
        preproc_form.addRow(input_scale_group)
        preproc_group.setLayout(preproc_form)
        layout.addRow(preproc_group)

    def createRunSysIdButton(self, layout):
        self.btn_run_sys_id = QPushButton("Run identification")
        self.btn_run_sys_id.clicked.connect(self.onSysIdClicked)
        self.btn_run_sys_id.setEnabled(False)
        layout.addRow(self.btn_run_sys_id)

    def createFindParamsButton(self, layout):
        self.btn_find_params = QPushButton("Find parameters")
        self.btn_find_params.clicked.connect(self.onFindParamsClicked)
        self.btn_find_params.setEnabled(False)
        layout.addRow(self.btn_find_params)

    def createFitWidget(self, layout):
        self.lbl_fit = QLabel("—")
        self.addHintedRow(
            layout,
            "Open-loop fit",
            self.lbl_fit,
            "Model driven by the logged input, compared with the logged output "
            "(100% is a perfect match).",
        )
        self.lbl_closed_loop_fit = QLabel("—")
        self.addHintedRow(
            layout,
            "Closed-loop fit",
            self.lbl_closed_loop_fit,
            "Logged reference replayed through the model and the controller "
            "gains flown in the log, compared with the logged output (100% is a "
            "perfect match).<br><br>"
            "Needs a preset with a <i>reference</i> and <i>gains</i> whose "
            "parameters are in the log, otherwise shows —.",
        )

    @staticmethod
    def addHintedRow(layout, name, value_label, hint):
        name_label = QLabel(f"{name} ⓘ")
        for label in (name_label, value_label):
            label.setToolTip(hint)
        layout.addRow(name_label, value_label)

    def createStabilityWidget(self, layout):
        self.lbl_stability = QLabel("—")
        self.btn_stabilize = QPushButton("Stabilize")
        self.btn_stabilize.setVisible(False)
        self.btn_stabilize.setToolTip(
            "Reflects unstable poles inside the unit circle (p → 1/p*)"
        )
        self.btn_stabilize.clicked.connect(self.stabilizeModel)
        stability_row = QHBoxLayout()
        stability_row.addWidget(self.lbl_stability)
        stability_row.addWidget(self.btn_stabilize)
        layout.addRow(QLabel("Stability"), stability_row)

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

    def onFindParamsClicked(self):
        if (
            hasattr(self, "_param_search_worker")
            and self._param_search_worker.isRunning()
        ):
            self._param_search_worker._cancel = True
            return
        self.btn_find_params.setText("Cancel (0%)")
        self._param_search_worker = ParamSearchWorker(
            self.u.copy(),
            self.y.copy(),
            self.t.copy(),
            self.dt,
            self.f_hp_spinbox.value(),
            self.f_lp_spinbox.value(),
            self.id_method_combo.currentText(),
        )
        self._param_search_worker.progress.connect(self.onParamSearchProgress)
        self._param_search_worker.finished.connect(self.onParamSearchFinished)
        self._param_search_worker.start()

    def onParamSearchProgress(self, current, total):
        self.btn_find_params.setText(f"Cancel ({100 * current // total}%)")

    def onParamSearchFinished(self, best_params, best_fit):
        self.line_edit_poles.setValue(best_params["n_poles"])
        self.line_edit_zeros.setValue(best_params["n_zeros"])
        self.line_edit_delays.setValue(best_params["delay"])
        self.btn_find_params.setText("Find parameters")
        self.btn_find_params.setEnabled(True)
        self.onSysIdClicked()

    def printImproperTfError(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Error")
        msg.setText("Transfer function must be proper, set Poles >= Zeros")
        msg.exec_()

    def createPidLayout(self):
        self.gain_slider = {}
        self.gain_edit = {form: {} for form in kGainFormLabels}

        def make_slider_callback(gain):
            return lambda: self.updateGainFromSlider(gain)

        def make_edit_callback(form, gain):
            return lambda: self.updateGainFromEdit(form, gain)

        def make_form_callback(form):
            return lambda checked: checked and self.setGainForm(form)

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

        self.gain_form_radio = {}
        for column, (form, label) in enumerate(kGainFormLabels.items(), start=2):
            self.gain_form_radio[form] = QRadioButton(label)
            layout_pid.addWidget(self.gain_form_radio[form], 0, column)

        for row, gain in enumerate(self.gains.keys(), start=1):
            label = gain if gain == "FF" else f"K{gain.lower()}"
            layout_pid.addWidget(QLabel(label), row, 0)

            self.gain_slider[gain] = DoubleSlider(Qt.Horizontal)
            self.gain_slider[gain].valueChanged.connect(make_slider_callback(gain))
            layout_pid.addWidget(self.gain_slider[gain], row, 1)

            for column, form in enumerate(kGainFormLabels, start=2):
                edit = QLineEdit()
                edit.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
                edit.setMinimumWidth(0)
                edit.setMinimumSize(0, 0)
                edit.setAlignment(Qt.AlignCenter)
                edit.textChanged.connect(make_edit_callback(form, gain))
                layout_pid.addWidget(edit, row, column)
                self.gain_edit[form][gain] = edit

        for form, radio in self.gain_form_radio.items():
            radio.toggled.connect(make_form_callback(form))
        self.gain_form_radio["ideal"].setChecked(True)

        return layout_pid

    def createStepInfoGroup(self):
        specs = {
            "rise_time": ("Rise time", 0.12, 0.01, 2.0, 0.01, "s"),
            "overshoot": ("Overshoot", 10.0, 0.0, 100.0, 0.5, "%"),
            "settling_time": ("Settling time", 0.4, 0.01, 5.0, 0.01, "s"),
        }
        group = QGroupBox("Step info")
        grid = QGridLayout()
        grid.addWidget(QLabel("Max"), 0, 1)
        grid.addWidget(QLabel("Measured"), 0, 2)
        for row, (key, (label, default, lo, hi, step, unit)) in enumerate(
            specs.items(), start=1
        ):
            sb = QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setSingleStep(step)
            sb.setDecimals(
                len(str(step).rstrip("0").split(".")[-1]) if "." in str(step) else 0
            )
            sb.setValue(default)
            sb.valueChanged.connect(self.onStepInfoChanged)
            self.step_info_spinbox[key] = sb

            measured_lbl = QLabel("—")
            self.step_info_measured_lbl[key] = measured_lbl

            grid.addWidget(QLabel(label + " (" + unit + ")"), row, 0)
            grid.addWidget(sb, row, 1)
            grid.addWidget(measured_lbl, row, 2)

        btn_plot_options = QPushButton("Plot options")
        btn_plot_options.setMenu(self.createPlotOptionsMenu(btn_plot_options))
        grid.addWidget(btn_plot_options, len(specs) + 1, 0, 1, 3)

        group.setLayout(grid)
        return group

    def createPlotOptionsMenu(self, parent):
        sim_specs = {
            "step_duration": (
                "Plot duration",
                self.step_duration,
                0.1,
                20.0,
                1.0,
                1,
                "s",
            ),
            "disturbance_time": (
                "Disturbance time",
                self.kDisturbanceTime,
                0.0,
                20.0,
                1.0,
                1,
                "s",
            ),
            "disturbance_amplitude": (
                "Disturbance ampl.",
                self.disturbance_amplitude,
                -1.0,
                1.0,
                0.1,
                2,
                "",
            ),
        }
        menu = QMenu(parent)
        content = QWidget(menu)
        form = QFormLayout(content)
        for key, (label, default, lo, hi, step, decimals, unit) in sim_specs.items():
            sb = QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setSingleStep(step)
            sb.setDecimals(decimals)
            sb.setValue(default)
            sb.valueChanged.connect(self.onStepSimChanged)
            self.step_sim_spinbox[key] = sb

            unit_str = " (" + unit + ")" if unit else ""
            form.addRow(label + unit_str, sb)

        action = QWidgetAction(menu)
        action.setDefaultWidget(content)
        menu.addAction(action)
        return menu

    def onStepSimChanged(self):
        self.step_duration = self.step_sim_spinbox["step_duration"].value()
        self.kDisturbanceTime = self.step_sim_spinbox["disturbance_time"].value()
        self.disturbance_amplitude = self.step_sim_spinbox[
            "disturbance_amplitude"
        ].value()
        self.updateClosedLoop()

    def onStepInfoChanged(self):
        for key in self.step_info:
            self.step_info[key] = self.step_info_spinbox[key].value()
        self.updateStepInfoEnvelope()

    def updateStepInfoEnvelope(self):
        if self.closed_loop_ax is None:
            return

        for patch in self.step_info_patches:
            patch.remove()
        self.step_info_patches = []

        ax = self.closed_loop_ax
        step_info = self.step_info
        kw = dict(color="#ff4444", linestyle="--", linewidth=1)

        y_limit = 1.0 + step_info["overshoot"] / 100.0
        p = ax.axhline(y_limit, **kw)
        self.step_info_patches.append(p)

        p = ax.plot(
            [step_info["settling_time"], step_info["settling_time"]], [0, 1.0], **kw
        )[0]
        self.step_info_patches.append(p)

        measured_map = {
            "rise_time": ("RiseTime", lambda v: f"{v:.3f}"),
            "overshoot": ("Overshoot", lambda v: f"{v:.1f}"),
            "settling_time": ("SettlingTime", lambda v: f"{v:.3f}"),
        }
        for key, (info_key, fmt) in measured_map.items():
            lbl = self.step_info_measured_lbl[key]
            if self.measured_step_info is None:
                lbl.setText("—")
                lbl.setStyleSheet("")
            else:
                measured = self.measured_step_info[info_key]
                lbl.setText(fmt(measured))
                color = thresholdColor(
                    measured, self.step_info[key], limit_is_minimum=False
                )
                lbl.setStyleSheet(f"color: {color}" if color else "")

        self.canvas.draw()

    def setGainForm(self, form):
        """Make the gains of form editable and driven by the sliders."""
        self.gain_form = form
        for edit_form, edits in self.gain_edit.items():
            for edit in edits.values():
                edit.setReadOnly(edit_form != form)
                edit.setFrame(edit_form == form)
        for gain, slider in self.gain_slider.items():
            minimum, maximum, step = kGainSliderRanges[form][gain]
            slider.setInterval(step)
            slider.setMinimum(minimum)
            slider.setMaximum(maximum)
        self.updateKIDSliders()

    def gainsInForm(self, form):
        return dict(self.gains) if form == "ideal" else parallelGains(self.gains)

    def setGainInActiveForm(self, gain, value):
        """Returns False when the gains have no ideal form (parallel Kp = 0)."""
        gains = self.gainsInForm(self.gain_form)
        gains[gain] = value
        ideal = idealGains(gains, self.gain_form)
        if ideal is None:
            return False
        self.gains = ideal
        return True

    def updateGainFromSlider(self, gain: str):
        if self.gain_slider[gain].hasFocus():
            self.setGainInActiveForm(gain, self.gain_slider[gain].value())
            self.showGains()
            if self.gain_slider[gain].isSliderDown():
                self.updateClosedLoop()

    def updateGainFromEdit(self, form, gain):
        edit = self.gain_edit[form][gain]
        if form != self.gain_form or not edit.hasFocus() or not isNumber(edit.text()):
            return
        value = float(edit.text())
        if self.setGainInActiveForm(gain, value):
            self.gain_slider[gain].setValue(value)
            # Rewriting the edit being typed in would move its cursor
            self.showGains(skip=edit)
            self.updateClosedLoop()

    def showGains(self, skip=None):
        for form, edits in self.gain_edit.items():
            gains = self.gainsInForm(form)
            for gain, edit in edits.items():
                if edit is not skip:
                    edit.setText(kGainFormat[form].format(gains[gain]))

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
        id.f_hp = self.f_hp_spinbox.value()
        id.f_lp = self.f_lp_spinbox.value()

        est = id.fit(self.u, self.y, method=self.id_method_combo.currentText())

        self.num = id.getNum()
        self.den = id.getDen()
        self.Gz = arx_transfer_function(self.num, self.den, self.dt)

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
        self.y_est = replayModel(self.Gz, self.sys_id_delays, self.t, self.u)
        self.t_est = self.t[: len(self.y_est)]
        showFit(self.lbl_fit, replayFit(self.y, self.y_est))
        self.updateClosedLoopFit()
        if self.plot_tabs.currentWidget() is self.validation_tab:
            self.updateValidation()

        self.plotInputOutput()
        self.checkStability()

    def updateClosedLoopFit(self):
        if self.reference is None or self.flown_gains is None:
            self.lbl_closed_loop_fit.setText("—")
            self.lbl_closed_loop_fit.setStyleSheet("")
            return
        fit = compute_closed_loop_fit(
            self.num,
            self.den,
            self.dt,
            self.sys_id_delays,
            self.flown_gains,
            self.t,
            self.reference,
            self.y,
            **self.flown_options,
        )
        if fit is None:
            self.lbl_closed_loop_fit.setText("unstable")
            self.lbl_closed_loop_fit.setStyleSheet("color: red")
        else:
            showFit(self.lbl_closed_loop_fit, fit)

    def checkStability(self):
        unstable = np.any(np.abs(self.Gz.poles()) > 1)
        if unstable:
            self.lbl_stability.setText("⚠ Unstable")
            self.lbl_stability.setStyleSheet("color: red")
            self.btn_stabilize.setVisible(True)
        else:
            self.lbl_stability.setText("✓ Stable")
            self.lbl_stability.setStyleSheet("color: green")
            self.btn_stabilize.setVisible(False)

    def stabilizeModel(self):
        poles = self.Gz.poles()
        stable_poles = np.where(np.abs(poles) > 1, 1.0 / np.conj(poles), poles)
        self.den = np.real(np.poly(stable_poles))
        self.Gz = arx_transfer_function(self.num, self.den, self.dt)
        self.updateTfDisplay(self.den[1:], self.num)
        self.plotPolesZeros()
        self.replayInputData()
        self.computeController()

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
        self.Gz = arx_transfer_function(self.num, self.den, self.dt)
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
        gains = self.gainsInForm(self.gain_form)
        for gain, slider in self.gain_slider.items():
            slider.setValue(gains[gain])
        self.showGains()

    def updateClosedLoop(self):
        if not self.is_system_identified:
            return
        loop = LoopModel(
            self.num,
            self.den,
            self.dt,
            self.sys_id_delays,
            self.gains,
            negate_output=self.negate_control_box.isChecked(),
            p_on_feedback=self.pid_no_zero_box.isChecked(),
        )

        t = np.arange(0, self.step_duration, self.dt)
        reference = np.ones_like(t)
        disturbance = np.where(
            t >= self.kDisturbanceTime, self.disturbance_amplitude, 0.0
        )
        self.plotClosedLoop(t, loop.simulate(t, reference, disturbance))

        stability_margins = loop.stabilityMargins()
        is_stable = loop.isStable()
        self.plotBode(
            loop.loop_gain, loop.reference_to_output, stability_margins, is_stable
        )
        self.plotNyquist(loop, stability_margins, is_stable)
        self.plotSensitivities(loop, is_stable)
        self.plotRootLocus(loop, stability_margins[0], is_stable)
        self.robustness_canvas.draw()

    def plotClosedLoop(self, t, y):
        # Compute metrics on pre-disturbance portion only
        mask = t < self.kDisturbanceTime
        try:
            self.measured_step_info = ctrl.step_info(
                y[mask], timepts=t[mask], final_output=1.0
            )
        except (IndexError, ValueError):
            self.measured_step_info = None

        step_ref = [1 if i > 0 else 0 for i in t]
        if self.closed_loop_ref is None:
            ax = self.figure.add_subplot(3, 3, 7)
            self.closed_loop_step_ref = ax.step(t, step_ref, "k--")[0]
            plot_ref = ax.plot(t, y)
            self.closed_loop_ref = plot_ref[0]
            self.closed_loop_ax = ax
            ax.set_title("Closed-loop step response")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude (rad/s)")
        else:
            self.closed_loop_step_ref.set_data(t, step_ref)
            self.closed_loop_ref.set_xdata(t)
            self.closed_loop_ref.set_ydata(y)
            self.closed_loop_ax.set_ylim(np.min(y), np.max([1.5, np.max(y)]))
        self.closed_loop_ax.set_xlim(t[0], t[-1])

        self.updateStepInfoEnvelope()

    def plotBode(self, open_loop, closed_loop, stability_margins, is_stable):
        gain_margin, phase_margin, _, phase_crossover, gain_crossover, _ = (
            stability_margins
        )
        gain_margin_db = 20 * np.log10(gain_margin)
        margins = {
            "gain": (
                f"Gain margin: {gain_margin_db:.2f}dB (@{phase_crossover / (2 * np.pi):.1f}Hz)",
                thresholdColor(
                    gain_margin_db, self.kMinGainMarginDb, limit_is_minimum=True
                ),
            ),
            "phase": (
                f"Phase margin: {phase_margin:.1f}deg (@{gain_crossover / (2 * np.pi):.1f}Hz)",
                thresholdColor(
                    phase_margin, self.kMinPhaseMarginDeg, limit_is_minimum=True
                ),
            ),
        }
        if not is_stable:
            margins = {"gain": (self.kUnstableLoopText, "red"), "phase": ("", None)}

        w = np.geomspace(0.1, np.pi / self.dt, 40).tolist()
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
            excludeAnnotationsFromLayout(ax)

            ax = self.figure.add_subplot(3, 3, (8, 9))
            plot_ref = ax.semilogx(f, phase_ol * 180 / np.pi, label="Open-loop")
            self.bode_plot_ref.append(plot_ref[0])
            plot_ref = ax.semilogx(f, phase_cl * 180 / np.pi, label="Closed-loop")
            self.bode_plot_ref.append(plot_ref[0])
            ax.set_ylim(-180, 180)

            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Phase (deg)")
            for row, key in enumerate(margins):
                self.margin_text_refs[key] = ax.text(
                    0.01,
                    0.9 - 0.12 * row,
                    "",
                    verticalalignment="top",
                    transform=ax.transAxes,
                )
            excludeAnnotationsFromLayout(ax)

        else:
            self.bode_plot_ref[0].set_xdata(f)
            mag_ol_db = 20 * np.log10(mag_ol)
            self.bode_plot_ref[0].set_ydata(mag_ol_db)
            self.bode_plot_ref[1].set_xdata(f)
            mag_cl_db = 20 * np.log10(mag_cl)
            self.bode_plot_ref[1].set_ydata(mag_cl_db)

            self.bode_plot_ref[2].set_xdata(f)
            self.bode_plot_ref[2].set_ydata(phase_ol * 180 / np.pi)
            self.bode_plot_ref[3].set_xdata(f)
            self.bode_plot_ref[3].set_ydata(phase_cl * 180 / np.pi)

        for key, (text, color) in margins.items():
            self.margin_text_refs[key].set_text(text)
            self.margin_text_refs[key].set_color(color or "black")

        self.canvas.draw()

    def plotNyquist(self, loop, stability_margins, is_stable):
        open_loop = loop.loop_gain
        w_nyquist = np.pi / self.dt
        w = np.geomspace(1e-2, w_nyquist, 2000)
        mag, phase, _ = ctrl.frequency_response(open_loop, omega=w)
        loop_response = mag * np.exp(1j * phase)

        if self.nyquist_ax is None:
            self.nyquist_ax = self.robustness_figure.add_subplot(2, 2, 1)
        ax = self.nyquist_ax
        ax.cla()

        unit_circle = np.exp(1j * np.linspace(0, 2 * np.pi, 200))
        ax.plot(unit_circle.real, unit_circle.imag, "k:", linewidth=0.8)
        forbidden_region = -1 + self.kMinModulusMargin * unit_circle
        ax.fill(
            forbidden_region.real,
            forbidden_region.imag,
            color="red",
            alpha=0.1,
            label=f"Modulus margin < {self.kMinModulusMargin}",
        )
        ax.axhline(0, color="k", linewidth=0.5)
        ax.axvline(0, color="k", linewidth=0.5)

        line = ax.plot(loop_response.real, loop_response.imag, label="Open-loop")[0]
        ax.plot(loop_response.real, -loop_response.imag, "--", color=line.get_color())
        ax.plot(-1, 0, "r+", markersize=12, markeredgewidth=2)

        if is_stable:
            margin_texts = self.annotateNyquistMargins(ax, loop, stability_margins)
        else:
            margin_texts = [(self.kUnstableLoopText, "red")]

        # A zero loop gain (Kp = 0) has no margin to show, and matplotlib
        # cannot lay out an empty box
        if margin_texts:
            lines = [
                TextArea(text, textprops=dict(color=color or "black", fontsize="small"))
                for text, color in margin_texts
            ]
            margin_box = AnchoredOffsetbox(
                loc="upper left",
                child=VPacker(children=lines, align="left", pad=0, sep=2),
                pad=0.3,
                borderpad=0.3,
            )
            margin_box.patch.set(alpha=0.8, edgecolor="none")
            ax.add_artist(margin_box)

        ax.set_xlim(-3, 1.5)
        ax.set_ylim(-2, 2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("Nyquist")
        ax.set_xlabel("Real")
        ax.set_ylabel("Imaginary")
        ax.legend(loc="lower right")
        excludeAnnotationsFromLayout(ax)

    def plotSensitivities(self, loop, is_stable):
        omega = np.geomspace(0.1, loop.nyquistFrequency(), 500)
        f = omega / (2 * np.pi)
        magnitudes_db = {
            key: 20 * np.log10(np.abs(response))
            for key, response in loop.sensitivities(omega).items()
        }

        if self.sensitivity_ax is None:
            self.sensitivity_ax = self.robustness_figure.add_subplot(2, 2, (2, 4))
        ax = self.sensitivity_ax
        ax.cla()

        labels = {
            "S": "S: sensitivity",
            "T": "T: complementary sensitivity",
            "PS": "PS: load disturbance → output",
            "CS": "CS: measurement noise → control",
            "CSF": "CSF: setpoint → control",
        }
        for key, label in labels.items():
            ax.semilogx(f, magnitudes_db[key], label=label)

        max_sensitivity_db = -20 * np.log10(self.kMinModulusMargin)
        ax.axhline(
            max_sensitivity_db,
            color="red",
            linestyle="--",
            linewidth=0.8,
            label=f"Max |S| ({max_sensitivity_db:.0f}dB)",
        )
        ax.axhline(0, color="k", linewidth=0.5)

        if is_stable:
            i_peak = np.argmax(magnitudes_db["S"])
            peak_db = magnitudes_db["S"][i_peak]
            color = thresholdColor(peak_db, max_sensitivity_db, limit_is_minimum=False)
            ax.plot(f[i_peak], peak_db, "o", color=color)
            ax.text(
                0.01,
                0.99,
                f"Peak sensitivity: {peak_db:.2f}dB (@{f[i_peak]:.1f}Hz)",
                color=color,
                verticalalignment="top",
                transform=ax.transAxes,
            )
        else:
            ax.text(
                0.01,
                0.99,
                self.kUnstableLoopText,
                color="red",
                verticalalignment="top",
                transform=ax.transAxes,
            )

        ax.set_xlim(f[0], f[-1])
        highest_db = max(np.max(m) for m in magnitudes_db.values())
        ax.set_ylim(-60, max(20, highest_db + 5))
        ax.grid(True, which="both", linewidth=0.3)
        ax.set_title("Sensitivity functions")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.legend(loc="lower left", fontsize="small")
        excludeAnnotationsFromLayout(ax)

    def plotRootLocus(self, loop, gain_margin, is_stable):
        if self.root_locus_ax is None:
            self.root_locus_ax = self.robustness_figure.add_subplot(2, 2, 3)
        ax = self.root_locus_ax
        ax.cla()
        drawZPlaneGrid(ax, loop.nyquistFrequency() / (2 * np.pi))

        kp = self.gains["P"]
        if kp != 0:
            max_gain_scale = 10.0
            if is_stable and np.isfinite(gain_margin):
                max_gain_scale = max(max_gain_scale, 2 * gain_margin)
            gain_scales = np.concatenate(
                ([0.0], np.geomspace(1e-3, max_gain_scale, 1000))
            )
            loci = loop.rootLocus(gain_scales)
            ax.plot(loci.real, loci.imag, color="C0", linewidth=1)
            ax.plot(
                loci[0].real, loci[0].imag, "x", color="C0", label="Open-loop poles"
            )
            zeros = ctrl.zeros(loop.loop_gain)
            ax.plot(
                zeros.real,
                zeros.imag,
                "o",
                color="C0",
                markerfacecolor="none",
                label="Open-loop zeros",
            )

            poles = loop.closedLoopPoles()
            ax.plot(
                poles.real,
                poles.imag,
                "s",
                color="green" if is_stable else "red",
                label=f"Kp = {kp:.3g}",
            )
            if is_stable and np.isfinite(gain_margin):
                poles = loop.closedLoopPoles(gain_margin)
                ax.plot(
                    poles.real,
                    poles.imag,
                    "D",
                    color="orange",
                    markerfacecolor="none",
                    label=f"Kp = {gain_margin * kp:.3g} (stability limit)",
                )
            ax.legend(loc="lower left", fontsize="small")
        else:
            ax.text(
                0.01,
                0.99,
                "Kp = 0: the loop is open",
                verticalalignment="top",
                transform=ax.transAxes,
            )

        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("Root locus (Kp)")
        ax.set_xlabel("Real")
        ax.set_ylabel("Imaginary")
        excludeAnnotationsFromLayout(ax)

    def annotateNyquistMargins(self, ax, loop, stability_margins):
        open_loop = loop.loop_gain
        (
            gain_margin,
            phase_margin,
            modulus_margin,
            phase_crossover,
            gain_crossover,
            modulus_margin_w,
        ) = stability_margins
        w_nyquist = np.pi / self.dt
        unit_circle = np.exp(1j * np.linspace(0, 2 * np.pi, 200))

        margin_texts = []

        if np.isfinite(gain_margin) and gain_margin > 0:
            gain_margin_db = 20 * np.log10(gain_margin)
            color = thresholdColor(
                gain_margin_db, self.kMinGainMarginDb, limit_is_minimum=True
            )
            ax.plot([-1, -1 / gain_margin], [0, 0], color=color, linewidth=2)
            ax.plot(-1 / gain_margin, 0, "o", color=color)
            margin_texts.append(
                (
                    f"Gain margin: {gain_margin_db:.2f}dB (@{phase_crossover / (2 * np.pi):.1f}Hz)",
                    color,
                )
            )

        if np.isfinite(phase_margin):
            color = thresholdColor(
                phase_margin, self.kMinPhaseMarginDeg, limit_is_minimum=True
            )
            # Arc from the critical point to the gain crossover on the unit circle
            arc_angle = np.linspace(np.pi, np.pi + np.deg2rad(phase_margin), 50)
            ax.plot(np.cos(arc_angle), np.sin(arc_angle), color=color, linewidth=2)
            crossover_point = np.exp(1j * arc_angle[-1])
            ax.plot(crossover_point.real, crossover_point.imag, "o", color=color)
            margin_texts.append(
                (
                    f"Phase margin: {phase_margin:.1f}deg (@{gain_crossover / (2 * np.pi):.1f}Hz)",
                    color,
                )
            )

        if np.isfinite(modulus_margin):
            color = thresholdColor(
                modulus_margin, self.kMinModulusMargin, limit_is_minimum=True
            )
            modulus_circle = -1 + modulus_margin * unit_circle
            ax.plot(
                modulus_circle.real,
                modulus_circle.imag,
                "--",
                color=color,
                linewidth=1,
            )
            closest_point = complex(
                ctrl.evalfr(
                    open_loop, np.exp(1j * min(modulus_margin_w, w_nyquist) * self.dt)
                )
            )
            ax.plot(
                [-1, closest_point.real],
                [0, closest_point.imag],
                color=color,
                linewidth=2,
            )
            ax.plot(closest_point.real, closest_point.imag, "o", color=color)
            margin_texts.append(
                (
                    f"Modulus margin: {modulus_margin:.2f} (@{modulus_margin_w / (2 * np.pi):.1f}Hz)",
                    color,
                )
            )

        delay_margin, delay_crossover = loop.delayMargin()
        if np.isfinite(delay_margin):
            delay_margin_samples = delay_margin / self.dt
            margin_texts.append(
                (
                    f"Delay margin: {delay_margin * 1e3:.1f}ms = {delay_margin_samples:.1f} samples (@{delay_crossover / (2 * np.pi):.1f}Hz)",
                    thresholdColor(
                        delay_margin_samples,
                        self.kMinDelayMarginSamples,
                        limit_is_minimum=True,
                    ),
                )
            )

        alpha, disk_gain_margin_db, disk_phase_margin_deg, disk_omega = (
            loop.diskMargins()
        )
        if alpha < 2:
            # The loop tolerates any gain f in the disk D(alpha) when L avoids
            # -1/f: a disk crossing the real axis at -a and -1/a
            a = (1 - alpha / 2) / (1 + alpha / 2)
            disk = -(a + 1 / a) / 2 + (1 / a - a) / 2 * unit_circle
            ax.plot(disk.real, disk.imag, ":", color="purple", linewidth=1.2)
            margin_texts.append(
                (
                    f"Disk margin: ±{disk_gain_margin_db:.2f}dB, ±{disk_phase_margin_deg:.1f}deg (@{disk_omega / (2 * np.pi):.1f}Hz)",
                    "purple",
                )
            )

        return margin_texts

    def scaleInput(self, u, true_airspeed):
        """Input scaled to trim airspeed, as selected in "Input scaling"."""
        if len(true_airspeed) != len(u):
            return u
        scale = 1
        scale_type = self.input_scale_choices[self.input_scale_combo.currentIndex()]
        if scale_type == "True airspeed":
            scale = np.array(true_airspeed) / self.trim_airspeed

        elif scale_type == "True airspeed^2":
            scale = (np.array(true_airspeed) / self.trim_airspeed) ** 2
        return u * scale

    def plotInputOutput(self, redraw=False):
        if len(self.true_airspeed) == len(self.input):
            self.u = self.scaleInput(self.input, self.true_airspeed)
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
            excludeAnnotationsFromLayout(ax)
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
        select = DataSelectionWindow(self.file_name, self.validation_windows)

        if select.exec_():
            self.reset()
            self.file_name = select.file_name
            self.validation_windows = select.validation_windows
            self.validation_data = select.validation_data
            self.t = select.t - select.t[0]
            self.input = select.u
            self.u = self.input
            self.y = select.y
            self.true_airspeed = select.v
            self.reference = select.r
            self.flown_gains = select.flown_gains
            self.flown_gain_form = select.flown_gain_form
            self.flown_options = select.flown_options or {}
            self.loadFlownController()
            trim_airspeed = select.getTrimAirspeed()

            if trim_airspeed is not None:
                self.line_edit_trim.setValue(trim_airspeed)

            self.refreshInputOutputData()
            self.btn_find_params.setEnabled(True)
            self.runIdentification()
            self.computeController()

    def loadFlownController(self):
        if "p_on_feedback" in self.flown_options:
            self.pid_no_zero_box.setChecked(self.flown_options["p_on_feedback"])
            self.negate_control_box.setChecked(self.flown_options["negate_output"])
        if self.flown_gains is not None:
            self.gains = dict(self.flown_gains)
        else:
            self.gains = dict(kMinimumGains)
        self.updateKIDSliders()
        if self.flown_gain_form is not None:
            self.gain_form_radio[self.flown_gain_form].setChecked(True)

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
        if self.reference is not None:
            self.reference = resample_interp(self.t, self.reference, t_new)

        if len(self.true_airspeed) > 0:
            self.true_airspeed = resample_interp(self.t, self.true_airspeed, t_new)

        self.t = t_new

    def validateOnWindow(self, t_log, u, y, v):
        """Replay of the identified model, with the same input scaling, on the
        data of a validation window."""
        t_log = t_log - t_log[0]
        t = np.arange(0, t_log[-1], self.dt)
        y = resample_interp(t_log, y, t)
        v = resample_interp(t_log, v, t) if len(v) else []
        u = self.scaleInput(resample_interp(t_log, u, t), v)
        y_est = replayModel(self.Gz, self.sys_id_delays, t, u)
        return {"t": t, "u": u, "y": y, "y_est": y_est, "fit": replayFit(y, y_est)}

    def updateValidation(self):
        """Validation tab: checks the identified model for overfitting by
        replaying it on each validation window, with the residual below. A fit
        close to the one of the identification window means it is not
        overfitted."""
        figure = self.validation_figure
        figure.clear()
        if not self.is_system_identified or not self.validation_windows:
            figure.text(
                0.5,
                0.5,
                "No validation window: select some in the log selection dialog",
                ha="center",
                va="center",
            )
            self.validation_canvas.draw()
            return

        # Each window: data, then residual
        grid = figure.add_gridspec(2 * len(self.validation_windows), 1)
        for row, ((t_start, t_stop), data) in enumerate(
            zip(self.validation_windows, self.validation_data)
        ):
            result = self.validateOnWindow(*data)
            t = result["t"]
            y_est = result["y_est"]
            fit = result["fit"]
            n = len(y_est)
            ax = figure.add_subplot(grid[2 * row, 0])
            ax.plot(t, result["u"], "C0", label="Input")
            ax.plot(t, result["y"], "C1", label="Output")
            if np.isfinite(fit):
                ax.plot(t[:n], y_est, "C2", label=f"Identified model, fit {fit:.1f}%")
            else:
                ax.plot([], [], "C2", label="Identified model: unstable, diverges")
            ax.set_title(f"{t_start:.1f}–{t_stop:.1f}s")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.legend(loc="upper left", fontsize="small")

            # Residual on the same detrended signals as the fit
            ax = figure.add_subplot(grid[2 * row + 1, 0], sharex=ax)
            if np.isfinite(fit):
                residual = detrend(result["y"][:n]) - detrend(y_est)
                ax.plot(
                    t[:n],
                    residual,
                    "C2",
                    linewidth=0.8,
                    label="RMS {:.3g}".format(np.sqrt(np.mean(residual**2))),
                )
                ax.legend(loc="upper left", fontsize="small")
            ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
            ax.set_title("Residual: output − model")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
        self.validation_canvas.draw()

    def onPlotTabChanged(self):
        if self.plot_tabs.currentWidget() is self.validation_tab:
            self.updateValidation()


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
