from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QPushButton, QLabel, QFormLayout, QRadioButton, QMessageBox, QFileDialog, QComboBox

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import SpanSelector

import numpy as np

from data_extractor import DataExtractor

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
        self.combo_u = QComboBox()
        self.combo_u.setEditable(True)
        self.combo_u.setInsertPolicy(QComboBox.NoInsert)
        self.combo_u.currentIndexChanged.connect(self.selectUData)
        in_out_group.addRow(QLabel("Input:"), self.combo_u)

        self.combo_y = QComboBox()
        self.combo_y.setEditable(True)
        self.combo_y.setInsertPolicy(QComboBox.NoInsert)
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

    def printRangeError(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Error")
        msg.setText("Range is invalid")
        msg.exec_()

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
            plot_refs = self.ax.plot([], [])
            self.input_ref = plot_refs[0]

            plot_refs = self.ax.plot([], [])
            self.output_ref = plot_refs[0]
            self.ax.autoscale(False)

            self.ax.set_title("Click and drag to select data range")
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Amplitude")
            self.ax.legend(["Input", "Output"])

            self.span = SpanSelector(self.ax, self.onselect, 'horizontal', useblit=False,
                                props=dict(alpha=0.2, facecolor='green'), interactive=True)

            self.canvas.mpl_connect('scroll_event', self.zoom_fun)
            self.canvas.draw()

    def plotU(self):
        self.input_ref.set_xdata(self.t)
        self.input_ref.set_ydata(self.u)
        self.resetXYLim()
        self.canvas.draw()

    def plotY(self):
        self.output_ref.set_xdata(self.t)
        self.output_ref.set_ydata(self.y)
        self.resetXYLim()
        self.canvas.draw()

    def resetXYLim(self):
        self.ax.set_xlim([self.t[0], self.t[-1]])

        if len(self.u) > 0 and len(self.y) > 0:
            self.ax.set_ylim([min([min(self.u), min(self.y)]), max([max(self.u), max(self.y)])])
        elif len(self.u) > 0:
            self.ax.set_ylim([min(self.u), max(self.u)])
        elif len(self.y) > 0:
            self.ax.set_ylim([min(self.y), max(self.y)])

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
