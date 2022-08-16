from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFormLayout, QRadioButton, QMessageBox

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import SpanSelector

import numpy as np

from data_extractor import getInputOutputData

class DataSelectionWindow(QDialog):
    def __init__(self, filename):
        QDialog.__init__(self)

        self.file_name = filename

        self.figure = plt.figure(1)
        self.canvas = FigureCanvas(self.figure)

        layout_v = QVBoxLayout()

        layout_v.addWidget(self.canvas)

        xyz_group = QHBoxLayout()
        r_x = QRadioButton("x")
        r_x.setChecked(True)
        r_y = QRadioButton("y")
        r_z = QRadioButton("z")
        xyz_group.addWidget(QLabel("Axis"))
        xyz_group.addWidget(r_x)
        xyz_group.addWidget(r_y)
        xyz_group.addWidget(r_z)
        r_x.clicked.connect(self.loadXData)
        r_y.clicked.connect(self.loadYData)
        r_z.clicked.connect(self.loadZData)

        layout_v.addLayout(xyz_group)

        btn_ok = QPushButton("Load selection")
        btn_ok.clicked.connect(self.loadLog)
        layout_v.addWidget(btn_ok)

        self.setLayout(layout_v)

        self.refreshInputOutputData()

    def loadLog(self):
        if self.t_stop > self.t_start:
            (self.t, self.u, self.y) = getInputOutputData(self.file_name, self.axis, self.t_start, self.t_stop)
            self.accept()
        else:
            self.printRangeError()

    def printRangeError(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Error")
        msg.setText("Range is invalid")
        msg.exec_()

    def loadXData(self):
        if self.file_name:
            self.refreshInputOutputData(0)

    def loadYData(self):
        if self.file_name:
            self.refreshInputOutputData(1)

    def loadZData(self):
        if self.file_name:
            self.refreshInputOutputData(2)

    def refreshInputOutputData(self, axis=0):
        if self.file_name:
            self.axis = axis
            (self.t, self.u, self.y) = getInputOutputData(self.file_name, axis)
            self.plotInputOutput(redraw=True)

    def plotInputOutput(self, redraw=False):
        self.figure.clear()
        self.ax = self.figure.add_subplot(1,1,1)
        self.ax.plot(self.t, self.u, self.t, self.y)
        self.ax.set_title("Click and drag to select data range")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.legend(["Input", "Output"])

        self.span = SpanSelector(self.ax, self.onselect, 'horizontal', useblit=False,
                            props=dict(alpha=0.2, facecolor='green'), interactive=True)

        self.t_start = self.t[0]
        self.t_stop = self.t[-1]

        self.canvas.mpl_connect('scroll_event', self.zoom_fun)

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
