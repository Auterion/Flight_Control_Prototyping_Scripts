import matplotlib.pyplot as plt
import numpy as np
from data_extractor import DataExtractor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import SpanSelector
from pid_analyse import plot_closed_loop_step_response
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)
from scipy import signal
from searchable_combo_box import SearchableComboBox


class DataSelectionWindow(QDialog):
    def __init__(self, filename):
        QDialog.__init__(self)

        self.preset_candidates = {
            "Rollrate": {
                "input": "vehicle_torque_setpoint/xyz[0].0",
                "output": "vehicle_angular_velocity/xyz[0].0",
                "input_legacy": "actuator_controls_0/control[0].0",
            },
            "Pitchrate": {
                "input": "vehicle_torque_setpoint/xyz[1].0",
                "output": "vehicle_angular_velocity/xyz[1].0",
                "input_legacy": "actuator_controls_0/control[1].0",
            },
            "Yawrate": {
                "input": "vehicle_torque_setpoint/xyz[2].0",
                "output": "vehicle_angular_velocity/xyz[2].0",
                "input_legacy": "actuator_controls_0/control[2].0",
            },
            "Rollrate(FW)": {
                "input": "vehicle_torque_setpoint/xyz[0].1",
                "output": "vehicle_angular_velocity/xyz[0].0",
                "input_legacy": "actuator_controls_1/control[0].0",
            },
            "Pitchrate(FW)": {
                "input": "vehicle_torque_setpoint/xyz[1].1",
                "output": "vehicle_angular_velocity/xyz[1].0",
                "input_legacy": "actuator_controls_1/control[1].0",
            },
            "Yawrate(FW)": {
                "input": "vehicle_torque_setpoint/xyz[2].1",
                "output": "vehicle_angular_velocity/xyz[2].0",
                "input_legacy": "actuator_controls_1/control[2].0",
            },
            "Rollrate(closed-loop)": {
                "input": "vehicle_rates_setpoint/roll.0",
                "output": "vehicle_angular_velocity/xyz[0].0",
            },
            "Pitchrate(closed-loop)": {
                "input": "vehicle_rates_setpoint/pitch.0",
                "output": "vehicle_angular_velocity/xyz[1].0",
            },
            "Yawrate(closed-loop)": {
                "input": "vehicle_rates_setpoint/yaw.0",
                "output": "vehicle_angular_velocity/xyz[2].0",
            },
        }

        self.presets = {}

        self.t = []
        self.u = []
        self.y = []
        self.t_start = None
        self.t_stop = None

        self.input_ref = None
        self.output_ref = None
        self.coherence_ref = None
        self.coherence_info_text = None
        self.figure = plt.figure(figsize=(8, 6), layout="constrained")
        self.canvas = FigureCanvas(self.figure)
        self.initPlots()

        layout_v = QVBoxLayout()

        top_group = QHBoxLayout()
        btn_browse = QPushButton("Browse files")
        btn_browse.clicked.connect(self.browseFiles)
        top_group.addWidget(btn_browse)

        in_out_group = QFormLayout()
        self.combo_preset = QComboBox()
        self.combo_preset.setEditable(False)

        self.combo_preset.currentIndexChanged.connect(self.selectPreset)
        in_out_group.addRow(QLabel("Preset:"), self.combo_preset)

        self.combo_u = SearchableComboBox()
        self.combo_u.currentIndexChanged.connect(self.selectUData)
        in_out_group.addRow(QLabel("Input:"), self.combo_u)

        self.combo_y = SearchableComboBox()
        self.combo_y.currentIndexChanged.connect(self.selectYData)
        in_out_group.addRow(QLabel("Output:"), self.combo_y)
        top_group.addLayout(in_out_group)

        layout_v.addLayout(top_group)
        layout_v.addWidget(self.canvas)

        self.label_warning = QLabel("")
        self.label_warning.setStyleSheet("color: red; font-weight: bold;")
        layout_v.addWidget(self.label_warning)

        btn_ok = QPushButton("Load selection")
        btn_ok.clicked.connect(self.loadSelection)
        layout_v.addWidget(btn_ok)

        pid_btn_ok = QPushButton("Analyse Current Tuning")
        pid_btn_ok.clicked.connect(self.plotPIDAnalysis)
        layout_v.addWidget(pid_btn_ok)

        self.setLayout(layout_v)

        if filename:
            self.file_name = filename
            self.openFile()

        else:
            self.browseFiles()

    def loadSelection(self):
        if (self.t_start is None and self.t_stop is None) or (
            self.t_stop > self.t_start
        ):
            (self.t, self.u, self.y, self.v) = self.data_extractor.getInputOutputData(
                self.topics[self.index_u],
                self.topics[self.index_y],
                self.t_start,
                self.t_stop,
            )
            self.accept()
        else:
            self.printRangeError()

    def browseFiles(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select ULog file", "", "ULog (*.ulg)", options=options
        )
        self.file_name = file_name
        self.openFile()

    def openFile(self):
        if self.file_name:
            self.data_extractor = DataExtractor(self.file_name)
            self.topics = self.data_extractor.get_topics_list()
            list_names = [
                f"{topic.topic_name}/{topic.variable_name}.{topic.instance}"
                for topic in self.topics
            ]
            self.combo_u.clear()
            self.combo_u.addItems(list_names)
            self.combo_y.clear()
            self.combo_y.addItems(list_names)

            # Trigger preset selection
            self.fillPresets()
            self.combo_preset.setCurrentIndex(0)
            self.selectPreset(0)

    def fillPresets(self):
        self.combo_preset.clear()
        self.presets = {}

        for candidate in self.preset_candidates:
            (index_u, index_y) = self.findInputOutputIndex(
                self.preset_candidates[candidate]
            )
            if index_u > -1 and index_y > -1:
                self.presets[candidate] = self.preset_candidates[candidate]

        self.combo_preset.addItems(list(self.presets.keys()))

    def printRangeError(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Error")
        msg.setText("Range is invalid")
        msg.exec_()

    def selectPreset(self, index):
        preset_key = list(self.presets.keys())[index]
        preset = self.presets[preset_key]
        (index_u, index_y) = self.findInputOutputIndex(preset)

        if index_u > -1:
            self.combo_u.setCurrentIndex(index_u)
        if index_y > -1:
            self.combo_y.setCurrentIndex(index_y)

    def findInputOutputIndex(self, preset):
        index_u = self.combo_u.findText(preset["input"])
        index_y = self.combo_u.findText(preset["output"])

        if index_u < 0 and "input_legacy" in preset:
            # Look for legacy topic
            index_u = self.combo_u.findText(preset["input_legacy"])

        if index_y < 0 and "output_legacy" in preset:
            # Look for legacy topic
            index_y = self.combo_u.findText(preset["output_legacy"])

        return (index_u, index_y)

    def selectUData(self, index):
        self.index_u = index
        (self.t, self.u) = self.data_extractor.getPreview(self.topics[index])
        self.plotU()

    def selectYData(self, index):
        self.index_y = index
        (self.t, self.y) = self.data_extractor.getPreview(self.topics[index])
        self.plotY()

    def getTrimAirspeed(self):
        return self.data_extractor.getTrimAirspeed()

    def initPlots(self):
        if self.input_ref is None:
            self.figure.clear()

            # --- Time series Axes (Top) ---
            self.ax = self.figure.add_subplot(2, 1, 1)
            color_in = "tab:blue"
            plot_refs = self.ax.plot([], [], color=color_in)
            self.input_ref = plot_refs[0]

            self.ax.autoscale(False)

            self.ax.set_title("Click and drag to select data range")
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Input", color=color_in)
            self.ax.tick_params(axis="y", labelcolor=color_in)

            color_out = "tab:orange"
            self.ax_out = self.ax.twinx()
            plot_refs = self.ax_out.plot([], [], color=color_out)
            self.output_ref = plot_refs[0]
            self.ax_out.set_ylabel("Output", color=color_out)
            self.ax_out.tick_params(axis="y", labelcolor=color_out)

            self.span = SpanSelector(
                self.ax_out,
                self.onselect,
                "horizontal",
                useblit=False,
                props=dict(alpha=0.2, facecolor="green"),
                interactive=True,
            )

            # --- Coherence Plot (Bottom) ---
            self.ax_coherence = self.figure.add_subplot(2, 1, 2)
            color_coherence = "tab:grey"
            plot_refs = self.ax_coherence.plot([], [], color=color_coherence)
            self.coherence_ref = plot_refs[0]
            self.ax_coherence.set_title("Coherence")
            self.ax_coherence.set_xlabel("Frequency (Hz)")
            self.ax_coherence.set_ylabel("Coherence")
            self.ax_coherence.set_xscale("log")

            self.canvas.mpl_connect("scroll_event", self.zoom_fun)
            self.canvas.draw()

    def plotU(self):
        self.input_ref.set_xdata(self.t)
        self.input_ref.set_ydata(self.u)
        self.ax.set_xlim([self.t[0], self.t[-1]])
        min_u = min(self.u)
        max_u = max(self.u)
        if min_u < max_u:
            self.ax.set_ylim([min_u, max_u])
        self.canvas.draw()

    def plotY(self):
        self.output_ref.set_xdata(self.t)
        self.output_ref.set_ydata(self.y)
        self.ax.set_xlim([self.t[0], self.t[-1]])
        min_y = min(self.y)
        max_y = max(self.y)
        if min_y < max_y:
            self.ax_out.set_ylim([min_y, max_y])
        self.canvas.draw()

    def onselect(self, xmin, xmax):
        indmin, indmax = np.searchsorted(self.t, (xmin, xmax))
        indmax = min(len(self.t) - 1, indmax)
        indmin = min(indmin, indmax)

        self.t_start = self.t[indmin]
        self.t_stop = self.t[indmax]
        self.ax.set_xlim(self.t_start - 1.0, self.t_stop + 1.0)
        self.canvas.draw()

        self.plotCoherence()

    def plotPIDAnalysis(self):
        if len(self.t) == 0 or len(self.u) == 0 or len(self.y) == 0:
            return

        if (
            self.t_start is not None
            and self.t_stop is not None
            and self.t_stop > self.t_start
        ):
            t_sel, u_sel, y_sel, _ = self.data_extractor.getInputOutputData(
                self.topics[self.index_u],
                self.topics[self.index_y],
                self.t_start,
                self.t_stop,
            )
        else:
            t_sel, u_sel, y_sel, _ = self.data_extractor.getInputOutputData(
                self.topics[self.index_u], self.topics[self.index_y]
            )

        # Create or reuse Step Response dialog
        if not hasattr(self, "step_dialog") or self.step_dialog is None:
            self.step_dialog = QDialog(self)
            self.step_dialog.setWindowTitle("Step Response")
            layout = QVBoxLayout(self.step_dialog)

            # Matplotlib figure + canvas
            self.step_fig, self.step_ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
            self.step_canvas = FigureCanvas(self.step_fig)
            layout.addWidget(self.step_canvas)

            self.step_dialog.setLayout(layout)

        else:
            # Clear previous plot if dialog already exists
            self.step_ax.clear()

        plot_closed_loop_step_response(u_sel, y_sel, t_sel, ax=self.step_ax)
        self.step_canvas.draw()

        self.step_dialog.show()
        self.step_dialog.raise_()


    def plotCoherence(self):
        if len(self.t) == 0 or len(self.u) == 0 or len(self.y) == 0:
            return

        # Use getInputOutputData with selected range
        if (
            self.t_start is not None
            and self.t_stop is not None
            and self.t_stop > self.t_start
        ):
            t_sel, u_sel, y_sel, _ = self.data_extractor.getInputOutputData(
                self.topics[self.index_u],
                self.topics[self.index_y],
                self.t_start,
                self.t_stop,
            )
        else:
            # If no range selected, just use full duration
            t_sel, u_sel, y_sel, _ = self.data_extractor.getInputOutputData(
                self.topics[self.index_u], self.topics[self.index_y]
            )

        num_samples = len(t_sel)
        duration = t_sel[-1] - t_sel[0]

        if num_samples < 256 or duration < 5:
            self.label_warning.setText(
                f"Increase the window size to at least 5 seconds and 256 samples. "
                f"Currently selected: {duration:.2f} seconds, {num_samples} samples."
            )
            self.label_warning.show()

            self.coherence_ref.set_xdata([])
            self.coherence_ref.set_ydata([])
            return
        else:
            self.label_warning.hide()

        # Estimate sampling frequency
        time_diffs = np.diff(t_sel)
        avg_time_diff = np.mean(time_diffs)
        if avg_time_diff == 0:
            return
        fs = 1 / avg_time_diff

        # Choose segment size
        nperseg = min(1024, num_samples // 4)

        # Compute coherence
        freq, Cuy = signal.coherence(u_sel, y_sel, fs, nperseg=nperseg)

        # Update coherence plot
        self.coherence_ref.set_xdata(freq)
        self.coherence_ref.set_ydata(Cuy)
        self.ax_coherence.set_xlim([0, 20])
        self.ax_coherence.set_ylim([0, 1])

        # Remove previous annotation if it exists
        if self.coherence_info_text is not None:
            self.coherence_info_text.remove()
            self.coherence_info_text = None

        freq_res = fs / nperseg
        info_text = (
            f"Samples: {num_samples}, Duration: {duration:.2f}s, "
            f"fs: {fs:.1f}Hz, nperseg: {nperseg}, Δf: {freq_res:.2f}Hz"
        )

        self.coherence_info_text = self.ax_coherence.text(
            0.98,
            0.02,
            info_text,
            ha="right",
            va="bottom",
            transform=self.ax_coherence.transAxes,
            fontsize=8,
            color="gray",
        )

        self.canvas.draw()

    def zoom_fun(self, event):
        base_scale = 1.1
        # get the current x and y limits
        cur_xlim = self.ax.get_xlim()
        cur_xrange = cur_xlim[1] - cur_xlim[0]
        xdata = event.xdata  # get event x location
        if xdata is None or xdata < cur_xlim[0] or xdata > cur_xlim[1]:
            return

        if event.button == "up":
            # deal with zoom in
            scale_factor = 1 / base_scale
        elif event.button == "down":
            # deal with zoom out
            scale_factor = base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
        # set new limits
        new_x_min = xdata - (xdata - cur_xlim[0]) * scale_factor
        new_x_max = xdata + (xdata - new_x_min) / (xdata - cur_xlim[0]) * (
            cur_xlim[1] - xdata
        )

        new_x_min = max(new_x_min, self.t[0] - 1.0)
        new_x_max = min(new_x_max, self.t[-1] + 1.0)
        self.ax.set_xlim([new_x_min, new_x_max])
        self.canvas.draw()
