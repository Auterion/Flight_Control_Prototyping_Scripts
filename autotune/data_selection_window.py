from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QPushButton, QLabel, QFormLayout, QRadioButton, QMessageBox, QFileDialog, QComboBox

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import SpanSelector

import numpy as np

from data_extractor import DataExtractor
from searchable_combo_box import SearchableComboBox

class DataSelectionWindow(QDialog):
    def __init__(self, filename):
        QDialog.__init__(self)

        self.t = []
        self.u = []
        self.y = []
        self.t_start = None
        self.t_stop = None

        self.input_ref = None
        self.output_ref = None
        self.figure = plt.figure(1)
        self.canvas = FigureCanvas(self.figure)
        self.initPlot()

        layout_v = QVBoxLayout()

        top_group = QHBoxLayout()
        btn_browse = QPushButton("Browse files")
        btn_browse.clicked.connect(self.browseFiles)
        top_group.addWidget(btn_browse)

        in_out_group = QFormLayout()
        self.combo_preset = QComboBox()
        self.combo_preset.setEditable(False)
        self.presets = ['Rollrate', 'Pitchrate', 'Yawrate']
        self.combo_preset.addItems(self.presets)
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

        btn_ok = QPushButton("Load selection")
        btn_ok.clicked.connect(self.loadSelection)
        layout_v.addWidget(btn_ok)

        self.setLayout(layout_v)

        if filename:
            self.file_name = filename
            self.openFile()

        else:
            self.browseFiles()

    def loadSelection(self):
        if (self.t_start is None and self.t_start is None) or (self.t_stop > self.t_start):
            (self.t, self.u, self.y, self.v) = self.data_extractor.getInputOutputData(self.topics[self.index_u], self.topics[self.index_y], self.t_start, self.t_stop)
            self.accept()
        else:
            self.printRangeError()

    def browseFiles(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self,"Select ULog file", "","ULog (*.ulg)", options=options)
        self.file_name = file_name
        self.openFile()

    def openFile(self):
        if self.file_name:
            self.data_extractor = DataExtractor(self.file_name)
            self.topics = self.data_extractor.get_topics_list()
            list_names = [f"{topic.topic_name}/{topic.variable_name}.{topic.instance}" for topic in self.topics]
            self.combo_u.clear()
            self.combo_u.addItems(list_names)
            self.combo_y.clear()
            self.combo_y.addItems(list_names)

            # Trigger preset selection
            self.combo_preset.setCurrentIndex(0)
            self.selectPreset(0)

    def printRangeError(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Error")
        msg.setText("Range is invalid")
        msg.exec_()

    def selectPreset(self, index):
        preset = self.presets[index]
        if preset == 'Rollrate':
            index_u = self.combo_u.findText("vehicle_torque_setpoint/xyz[0].0")
            index_y = self.combo_u.findText("vehicle_angular_velocity/xyz[0].0")

            if index_u < 0:
                # Look for legacy topic
                index_u = self.combo_u.findText("actuator_controls_0/control[0].0")

        elif preset == 'Pitchrate':
            index_u = self.combo_u.findText("vehicle_torque_setpoint/xyz[1].0")
            index_y = self.combo_u.findText("vehicle_angular_velocity/xyz[1].0")

            if index_u < 0:
                # Look for legacy topic
                index_u = self.combo_u.findText("actuator_controls_0/control[1].0")

        elif preset == 'Yawrate':
            index_u = self.combo_u.findText("vehicle_torque_setpoint/xyz[2].0")
            index_y = self.combo_u.findText("vehicle_angular_velocity/xyz[2].0")

            if index_u < 0:
                # Look for legacy topic
                index_u = self.combo_u.findText("actuator_controls_0/control[2].0")

        if index_u > -1:
            self.combo_u.setCurrentIndex(index_u)
        if index_y > -1:
            self.combo_y.setCurrentIndex(index_y)

    def selectUData(self, index):
        self.index_u = index
        (self.t, self.u) = self.data_extractor.getPreview(self.topics[index])
        self.plotU()

    def selectYData(self, index):
        self.index_y = index
        (self.t, self.y) = self.data_extractor.getPreview(self.topics[index])
        self.plotY()

    def initPlot(self):
        if self.input_ref is None:
            self.figure.clear()
            self.ax = self.figure.add_subplot(1,1,1)
            color_in = 'tab:blue'
            plot_refs = self.ax.plot([], [], color=color_in)
            self.input_ref = plot_refs[0]

            self.ax.autoscale(False)

            self.ax.set_title("Click and drag to select data range")
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Input", color=color_in)
            self.ax.tick_params(axis='y', labelcolor=color_in)

            color_out = 'tab:orange'
            self.ax_out = self.ax.twinx()
            plot_refs = self.ax_out.plot([], [], color=color_out)
            self.output_ref = plot_refs[0]
            self.ax_out.set_ylabel("Output", color=color_out)
            self.ax_out.tick_params(axis='y', labelcolor=color_out)

            self.span = SpanSelector(self.ax_out, self.onselect, 'horizontal', useblit=False,
                                props=dict(alpha=0.2, facecolor='green'), interactive=True)

            self.canvas.mpl_connect('scroll_event', self.zoom_fun)
            self.canvas.draw()

    def plotU(self):
        self.input_ref.set_xdata(self.t)
        self.input_ref.set_ydata(self.u)
        self.ax.set_xlim([self.t[0], self.t[-1]])
        self.ax.set_ylim([min(self.u), max(self.u)])
        self.canvas.draw()

    def plotY(self):
        self.output_ref.set_xdata(self.t)
        self.output_ref.set_ydata(self.y)
        self.ax.set_xlim([self.t[0], self.t[-1]])
        self.ax_out.set_ylim([min(self.y), max(self.y)])
        self.canvas.draw()

    def onselect(self, xmin, xmax):
        indmin, indmax = np.searchsorted(self.t, (xmin, xmax))
        indmax = min(len(self.t) - 1, indmax)
        indmin = min(indmin, indmax)

        self.t_start = self.t[indmin]
        self.t_stop = self.t[indmax]
        self.ax.set_xlim(self.t_start - 1.0, self.t_stop + 1.0)
        self.canvas.draw()

    def zoom_fun(self, event):
        base_scale = 1.1
        # get the current x and y limits
        cur_xlim = self.ax.get_xlim()
        cur_xrange = cur_xlim[1] - cur_xlim[0]
        xdata = event.xdata # get event x location
        if xdata is None or xdata < cur_xlim[0] or xdata > cur_xlim[1]:
            return

        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1/base_scale
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
        # set new limits
        new_x_min = xdata - (xdata - cur_xlim[0])*scale_factor
        new_x_max = xdata + (xdata - new_x_min) / (xdata - cur_xlim[0]) * (cur_xlim[1] - xdata)

        new_x_min = max(new_x_min, self.t[0] - 1.0)
        new_x_max = min(new_x_max, self.t[-1] + 1.0)
        self.ax.set_xlim([new_x_min, new_x_max])
        self.canvas.draw()
