import numpy as np
import plot_style as ps
import pyqtgraph as pg
from data_extractor import DataExtractor
from loop_model import idealGains
from pid_analyse_window import PIDAnalyseWindow
from preset_dialogs import PresetEditDialog
from presets import load_presets, save_presets
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)
from scipy import signal
from searchable_combo_box import SearchableComboBox


class SpanSelectViewBox(ps.ResettableViewBox):
    """Left drag draws a time range.

    The other buttons keep the pyqtgraph behavior: the middle button pans and
    the right one zooms.
    """

    sigSpanDragged = pyqtSignal(float, float, bool)  # start, stop, finished

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() != Qt.LeftButton or axis is not None:
            super().mouseDragEvent(ev, axis)
            return
        ev.accept()
        start = self.mapSceneToView(ev.buttonDownScenePos()).x()
        stop = self.mapSceneToView(ev.scenePos()).x()
        self.sigSpanDragged.emit(min(start, stop), max(start, stop), ev.isFinish())


class DataSelectionWindow(QDialog):
    def __init__(self, filename):
        QDialog.__init__(self)

        self.preset_candidates = load_presets()

        self.presets = {}
        self.topic_names = []

        self.t = []
        self.u = []
        self.y = []
        self.r = None
        self.flown_gains = None
        self.flown_gain_form = None
        self.flown_options = None
        self.t_start = None
        self.t_stop = None

        self.plots = self.createPlots()

        layout_v = QVBoxLayout()

        top_group = QHBoxLayout()
        btn_browse = QPushButton("Browse files")
        btn_browse.clicked.connect(self.browseFiles)
        top_group.addWidget(btn_browse)

        in_out_group = QFormLayout()
        self.combo_preset = QComboBox()
        self.combo_preset.setEditable(False)

        self.combo_preset.currentIndexChanged.connect(self.selectPreset)
        preset_row = QHBoxLayout()
        preset_row.addWidget(self.combo_preset)
        self.btn_add_preset = QPushButton("Add")
        self.btn_add_preset.clicked.connect(self.addPreset)
        preset_row.addWidget(self.btn_add_preset)
        self.btn_edit_preset = QPushButton("Edit")
        self.btn_edit_preset.clicked.connect(self.editPreset)
        preset_row.addWidget(self.btn_edit_preset)
        in_out_group.addRow(QLabel("Preset:"), preset_row)

        self.combo_u = SearchableComboBox()
        self.combo_u.currentIndexChanged.connect(self.selectUData)
        in_out_group.addRow(QLabel("Input:"), self.combo_u)

        self.combo_y = SearchableComboBox()
        self.combo_y.currentIndexChanged.connect(self.selectYData)
        in_out_group.addRow(QLabel("Output:"), self.combo_y)
        top_group.addLayout(in_out_group)

        layout_v.addLayout(top_group)
        layout_v.addWidget(self.plots)

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
            (self.r, self.flown_gains, self.flown_gain_form, self.flown_options) = (
                self.getFlownController()
            )
            self.accept()
        else:
            self.printRangeError()

    def getFlownController(self):
        """Reference, flown gains and PID options of the selected preset's loop.

        Returns (reference, gains, gain_form, options): gains in ideal form,
        gain_form the form of the preset's parameters and options as LoopModel
        keyword arguments. Each is None when the preset does not provide it or
        the selected signals no longer match the preset.
        """
        preset = self.preset_candidates.get(self.combo_preset.currentText())
        if preset is None or self.findInputOutputIndex(preset) != (
            self.index_u,
            self.index_y,
        ):
            return (None, None, None, None)

        options = {
            "p_on_feedback": preset.get("pi_no_zero", False),
            "negate_output": preset.get("negate_output", False),
        }

        reference = None
        index_r = self.combo_u.findText(preset.get("reference", ""))
        if "reference" in preset and index_r > -1:
            reference = self.data_extractor.getAlignedData(self.topics[index_r], self.t)

        return (
            reference,
            self.getFlownGains(preset),
            preset.get("gain_form", "parallel"),
            options,
        )

    def getFlownGains(self, preset):
        if "gains" not in preset:
            return None
        values = {
            gain: self.data_extractor.getParameter(param, self.t[0])
            for gain, param in preset["gains"].items()
        }
        if None in values.values():
            return None
        return idealGains(values, preset.get("gain_form", "parallel"))

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
            self.topic_names = list_names
            self.combo_u.clear()
            self.combo_u.addItems(list_names)
            self.combo_y.clear()
            self.combo_y.addItems(list_names)

            # Show all presets, then select the first one that matches the
            # available topics. Presets that don't match are still listed (and
            # editable) but greyed out and won't auto-fill the input/output.
            self.fillPresets()
            for i in range(self.combo_preset.count()):
                if self.combo_preset.itemText(i) in self.presets:
                    self.combo_preset.setCurrentIndex(i)
                    self.selectPreset(i)
                    break

    def fillPresets(self):
        self.combo_preset.blockSignals(True)
        self.combo_preset.clear()
        self.presets = {}

        # List every preset so the user can see and edit all of them. Track
        # which ones match the loaded log's topics in self.presets.
        for index, candidate in enumerate(self.preset_candidates):
            self.combo_preset.addItem(candidate)
            (index_u, index_y) = self.findInputOutputIndex(
                self.preset_candidates[candidate]
            )
            if index_u > -1 and index_y > -1:
                self.presets[candidate] = self.preset_candidates[candidate]
            else:
                # Grey out presets that don't match the current log.
                item = self.combo_preset.model().item(index)
                if item is not None:
                    item.setForeground(QBrush(QColor("gray")))

        self.combo_preset.blockSignals(False)

    def printRangeError(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Error")
        msg.setText("Range is invalid")
        msg.exec_()

    def selectPreset(self, index):
        name = self.combo_preset.itemText(index)
        if name not in self.preset_candidates:
            return
        preset = self.preset_candidates[name]
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

    def addPreset(self):
        self._openPresetDialog(create=True)

    def editPreset(self):
        self._openPresetDialog(create=False)

    def _openPresetDialog(self, create):
        if not self.topic_names:
            QMessageBox.information(
                self, "No log loaded", "Open a log file before editing presets."
            )
            return

        # The currently selected preset (from the topic-matched subset shown in
        # the combo); may be empty if no preset matched the log.
        selected_name = self.combo_preset.currentText()
        if selected_name not in self.preset_candidates:
            selected_name = None

        if not create and selected_name is None:
            QMessageBox.information(
                self, "No preset selected", "Select a preset to edit first."
            )
            return

        dialog = PresetEditDialog(
            self,
            create,
            selected_name,
            self.preset_candidates,
            self.topic_names,
            self.combo_u.currentText(),
            self.combo_y.currentText(),
            self.data_extractor.getParameterList(),
        )
        dialog.exec_()

        if dialog.result_action == "save":
            if dialog.remove_name and dialog.remove_name != dialog.name:
                self.preset_candidates.pop(dialog.remove_name, None)
            self.preset_candidates[dialog.name] = dialog.preset
            target = dialog.name
        elif dialog.result_action == "delete":
            self.preset_candidates.pop(dialog.remove_name, None)
            target = None
        else:
            return

        save_presets(self.preset_candidates)
        self.fillPresets()

        # Reselect the affected preset if it is still present and matches.
        if target is not None:
            index = self.combo_preset.findText(target)
            if index > -1:
                self.combo_preset.setCurrentIndex(index)

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

    def createPlots(self):
        layout = pg.GraphicsLayoutWidget()
        layout.setMinimumSize(800, 600)
        self.hover_readouts = []

        self.input_plot = ps.addPlot(
            layout,
            0,
            0,
            view_box=SpanSelectViewBox(),
            title="Drag to select the data range",
            ylabel="Input",
        )
        self.output_plot = ps.addPlot(
            layout,
            1,
            0,
            view_box=SpanSelectViewBox(),
            xlabel="Time (s)",
            ylabel="Output",
        )
        self.output_plot.setXLink(self.input_plot)
        self.input_curve = self.input_plot.plot(pen=ps.pen(ps.BLUE))
        self.output_curve = self.output_plot.plot(pen=ps.pen(ps.ORANGE))

        self.selection_regions = []
        for plot in (self.input_plot, self.output_plot):
            ps.xOnlyMouse(plot)
            plot.getViewBox().sigSpanDragged.connect(self.onSpanDragged)
            region = pg.LinearRegionItem(
                brush=ps.color(ps.GREEN, 0.15),
                hoverBrush=ps.color(ps.GREEN, 0.25),
                pen=ps.pen(ps.GREEN),
            )
            region.hide()
            region.sigRegionChanged.connect(self.onRegionChanged)
            region.sigRegionChangeFinished.connect(self.onRegionChangeFinished)
            plot.addItem(region, ignoreBounds=True)
            self.selection_regions.append(region)
            self.hover_readouts.append(ps.HoverReadout(plot, "t = {:.3f} s"))

        self.coherence_plot = ps.addPlot(
            layout,
            2,
            0,
            title="Coherence",
            xlabel="Frequency (Hz)",
            ylabel="Coherence",
        )
        self.coherence_plot.setLogMode(x=True)
        self.coherence_plot.setMouseEnabled(x=True, y=False)
        self.coherence_curve = self.coherence_plot.plot(pen=ps.pen(ps.INK_SECONDARY))
        self.coherence_info_text = ps.CornerText(self.coherence_plot, "bottom-right")
        self.coherence_plot.getViewBox().setDefaultRange(
            (np.log10(0.1), np.log10(20)), (0, 1)
        )
        self.hover_readouts.append(
            ps.HoverReadout(
                self.coherence_plot, "f = {:.3g} Hz", "{:.2f}", corner="top-right"
            )
        )
        return layout

    def plotU(self):
        self.plotTimeSeries(self.input_plot, self.input_curve, self.u)

    def plotY(self):
        self.plotTimeSeries(self.output_plot, self.output_curve, self.y)

    def plotTimeSeries(self, plot, curve, values):
        curve.setData(self.t, values)
        plot.setLimits(xMin=self.t[0] - 1.0, xMax=self.t[-1] + 1.0)
        plot.setXRange(self.t[0], self.t[-1], padding=0)

    def setSelectionRegion(self, t_start, t_stop):
        for region in self.selection_regions:
            region.blockSignals(True)
            region.setRegion((t_start, t_stop))
            region.blockSignals(False)
            region.show()

    def onSpanDragged(self, t_start, t_stop, finished):
        self.setSelectionRegion(t_start, t_stop)
        if finished:
            self.onselect(t_start, t_stop)

    def onRegionChanged(self, changed_region):
        t_start, t_stop = changed_region.getRegion()
        for region in self.selection_regions:
            if region is not changed_region:
                region.blockSignals(True)
                region.setRegion((t_start, t_stop))
                region.blockSignals(False)

    def onRegionChangeFinished(self, region):
        self.onselect(*region.getRegion())

    def onselect(self, xmin, xmax):
        indmin, indmax = np.searchsorted(self.t, (xmin, xmax))
        indmax = min(len(self.t) - 1, indmax)
        indmin = min(indmin, indmax)

        self.t_start = self.t[indmin]
        self.t_stop = self.t[indmax]
        self.setSelectionRegion(self.t_start, self.t_stop)
        self.input_plot.setXRange(self.t_start - 1.0, self.t_stop + 1.0, padding=0)

        self.plotCoherence()

    def plotPIDAnalysis(self):
        if len(self.t) == 0 or len(self.u) == 0 or len(self.y) == 0:
            return

        # Get input/output data
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

        # Create analysis window if not open yet
        if not hasattr(self, "pid_window") or self.pid_window is None:
            self.pid_window = PIDAnalyseWindow(self)

        # Show
        self.pid_window.show()
        self.pid_window.raise_()

        # Update plot
        self.pid_window.generate_step_response(u_sel, y_sel, t_sel)

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

            self.coherence_curve.setData([], [])
            self.coherence_info_text.setLines([])
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

        self.coherence_curve.setData(freq, Cuy)

        freq_res = fs / nperseg
        info_text = (
            f"Samples: {num_samples}, Duration: {duration:.2f}s, "
            f"fs: {fs:.1f}Hz, nperseg: {nperseg}, Δf: {freq_res:.2f}Hz"
        )
        self.coherence_info_text.setLines([(info_text, ps.MUTED)])
