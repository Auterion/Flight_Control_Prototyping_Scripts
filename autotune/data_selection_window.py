from dataclasses import dataclass

import numpy as np
import plot_style as ps
import pyqtgraph as pg
from data_extractor import DataExtractor
from loop_model import idealGains
from pid_analyse_window import PIDAnalyseWindow
from preset_dialogs import PresetEditDialog
from presets import load_presets, save_presets
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont
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

# Coherence is averaged over num_samples // 4 long segments: below, the
# frequency resolution fs / nperseg is too coarse for a rate loop
kMinCoherenceSamples = 256
# Coherence above which the frequency response is accurate enough to
# identify from (Tischler and Remple, Aircraft and Rotorcraft System
# Identification)
kMinGoodCoherence = 0.6

kMissingSamplesWarning = (
    "The selected range contains missing (NaN) samples, e.g. while the motor is "
    "disarmed. Select a range where the input and output are logged."
)


kEstimation = "estimation"
kValidation = "validation"
kRangeColors = {kEstimation: ps.GREEN, kValidation: ps.VIOLET}
kValidationRangeHint = (
    "<b>Why a validation range?</b><br>"
    "On the data it was fitted to, a model with more parameters always fits "
    "better: it also fits the noise. On another maneuver, a model that fits "
    "the noise predicts worse, which shows the model to keep. The fits, the "
    "residuals and <i>Find parameters</i> then judge the model on the "
    "validation range.<br><br>"
    "<b>How to choose the ranges</b><br>"
    "• <b>Estimation:</b> a long range of rich excitation, covering the "
    "frequencies of the loop, e.g. a sweep or the PX4 autotune "
    "excitation.<br>"
    "• <b>Validation:</b> a shorter, separate range in the same flight "
    "condition, not overlapping the estimation one, ideally with another "
    "input, e.g. steps or pilot inputs.<br><br>"
    "The validation range is optional: without it, the model is judged on "
    "the estimation range. The ✕ in the corner of a range removes it."
)


def hasMissingSamples(*signals):
    return any(np.isnan(values).any() for values in signals)


def isValidRange(t_start, t_stop):
    if t_start is None and t_stop is None:
        return True
    return t_start is not None and t_stop is not None and t_stop > t_start


@dataclass
class ValidationSelection:
    """Data of the validation range, in log time.

    r and flown_gains are the reference and the gains flown during this
    range, None as for the estimation range.
    """

    t: np.ndarray
    u: np.ndarray
    y: np.ndarray
    v: np.ndarray
    r: np.ndarray | None
    flown_gains: dict | None


class RangeRemoveButton(pg.TextItem):
    """✕ in the top right corner of a selected range, a click removes it."""

    sigClicked = pyqtSignal()

    def __init__(self, color_):
        super().__init__("✕", color=color_, anchor=(1, 0))
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.setFont(font)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Remove this range")
        # Above the range, which would take the click
        self.setZValue(100)

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            ev.accept()
            self.sigClicked.emit()


class SpanSelectViewBox(ps.ResettableViewBox):
    """Left and right drags draw a time range.

    The middle button keeps the pyqtgraph behavior: it pans.
    """

    sigSpanDragged = pyqtSignal(
        float, float, bool, int
    )  # start, stop, finished, button

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() not in (Qt.LeftButton, Qt.RightButton) or axis is not None:
            super().mouseDragEvent(ev, axis)
            return
        ev.accept()
        start = self.mapSceneToView(ev.buttonDownScenePos()).x()
        stop = self.mapSceneToView(ev.scenePos()).x()
        self.sigSpanDragged.emit(
            min(start, stop), max(start, stop), ev.isFinish(), int(ev.button())
        )


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
        self.flown_gain_scale = None
        self.flown_options = None
        self.t_start = None
        self.t_stop = None
        # The model is judged on data it was not fitted to. None validates on
        # the estimation range
        self.validation_range = None
        self.validation = None

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
        if not isValidRange(self.t_start, self.t_stop):
            self.printRangeError()
            return
        (t, u, y, v) = self.getInputOutputData(self.t_start, self.t_stop)
        if hasMissingSamples(u, y):
            self.printMissingSamplesError()
            return

        validation = None
        if self.validation_range is not None:
            (t_v, u_v, y_v, v_v) = self.getInputOutputData(*self.validation_range)
            if hasMissingSamples(u_v, y_v):
                self.printMissingSamplesError()
                return
            r_v, flown_gains_v, *_ = self.getFlownController(t_v)
            validation = ValidationSelection(t_v, u_v, y_v, v_v, r_v, flown_gains_v)

        (self.t, self.u, self.y, self.v) = (t, u, y, v)
        self.validation = validation
        (
            self.r,
            self.flown_gains,
            self.flown_gain_form,
            self.flown_gain_scale,
            self.flown_options,
        ) = self.getFlownController(self.t)
        self.accept()

    def getInputOutputData(self, t_start=None, t_stop=None):
        """Input, output and airspeed of the range, of the whole log when
        t_start and t_stop are None."""
        return self.data_extractor.getInputOutputData(
            self.topics[self.index_u], self.topics[self.index_y], t_start, t_stop
        )

    def getFlownController(self, t):
        """Reference, flown gains and PID options of the selected preset's loop
        over the time base t.

        Returns (reference, gains, gain_form, gain_scale, options): gains in
        ideal form, gain_form the form of the preset's parameters, gain_scale
        the gain of a parameter unit and options as LoopModel keyword
        arguments. Each is None when the preset does not provide it or
        the selected signals no longer match the preset.
        """
        preset = self.preset_candidates.get(self.combo_preset.currentText())
        if preset is None or self.findInputOutputIndex(preset) != (
            self.index_u,
            self.index_y,
        ):
            return (None, None, None, None, None)

        options = {
            "p_on_feedback": preset.get("pi_no_zero", False),
            "negate_output": preset.get("negate_output", False),
        }

        reference = None
        index_r = self.combo_u.findText(preset.get("reference", ""))
        if "reference" in preset and index_r > -1:
            reference = self.data_extractor.getAlignedData(self.topics[index_r], t)

        return (
            reference,
            self.getFlownGains(preset, t[0]),
            preset.get("gain_form", "parallel"),
            preset.get("gain_scale", 1.0),
            options,
        )

    def getFlownGains(self, preset, t):
        if "gains" not in preset:
            return None
        values = {
            gain: self.data_extractor.getParameter(param, t)
            for gain, param in preset["gains"].items()
        }
        if None in values.values():
            return None
        scale = preset.get("gain_scale", 1.0)
        values = {gain: value * scale for gain, value in values.items()}
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

    def printMissingSamplesError(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Error")
        msg.setText(kMissingSamplesWarning)
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
            title=(
                f"<span style='color:{kRangeColors[kEstimation]}'>Left drag: "
                "estimation range</span>, "
                f"<span style='color:{kRangeColors[kValidation]}'>right drag: "
                "validation range</span>"
            ),
            info=kValidationRangeHint,
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

        self.selection_regions = {kEstimation: [], kValidation: []}
        self.range_remove_buttons = {kEstimation: [], kValidation: []}
        for plot in (self.input_plot, self.output_plot):
            ps.xOnlyMouse(plot)
            plot.getViewBox().sigSpanDragged.connect(self.onSpanDragged)
            for kind, regions in self.selection_regions.items():
                remove_button = RangeRemoveButton(kRangeColors[kind])
                remove_button.hide()
                remove_button.sigClicked.connect(
                    lambda kind=kind: self.removeRange(kind)
                )
                # Ignored by the autorange, which would otherwise grow to make
                # room for the button it moves to the top of the view
                plot.addItem(remove_button, ignoreBounds=True)
                self.range_remove_buttons[kind].append(remove_button)
                region = pg.LinearRegionItem(
                    brush=ps.color(kRangeColors[kind], 0.15),
                    hoverBrush=ps.color(kRangeColors[kind], 0.25),
                    pen=ps.pen(kRangeColors[kind]),
                )
                region.hide()
                region.sigRegionChanged.connect(
                    lambda changed, kind=kind: self.onRegionChanged(kind, changed)
                )
                region.sigRegionChanged.connect(
                    lambda changed, button=remove_button: self.placeRemoveButton(
                        button, changed
                    )
                )
                plot.getViewBox().sigRangeChanged.connect(
                    lambda *_, button=remove_button, region=region: (
                        self.placeRemoveButton(button, region)
                    )
                )
                region.sigRegionChangeFinished.connect(
                    lambda changed, kind=kind: self.onselect(kind, *changed.getRegion())
                )
                plot.addItem(region, ignoreBounds=True)
                regions.append(region)
            self.hover_readouts.append(ps.HoverReadout(plot, "t = {:.3f} s"))

        self.coherence_plot = ps.addPlot(
            layout,
            2,
            0,
            title="Coherence",
            xlabel="Frequency (Hz)",
            ylabel="Coherence γ²",
        )
        self.coherence_plot.setLogMode(x=True)
        self.coherence_threshold_line = ps.addReferenceLine(
            self.coherence_plot, kMinGoodCoherence, angle=0, color_=ps.MUTED, style="--"
        )
        pg.InfLineLabel(
            self.coherence_threshold_line,
            text=f"good ≥ {kMinGoodCoherence}",
            position=0.03,
            anchors=[(0, 1), (0, 1)],
            color=ps.MUTED,
        )
        self.coherence_plot.setMouseEnabled(x=True, y=False)
        self.coherence_legend = ps.addLegend(self.coherence_plot, corner="bottom-left")
        self.coherence_curves = {
            kind: self.coherence_plot.plot(pen=ps.pen(kRangeColors[kind], 1.5))
            for kind in (kEstimation, kValidation)
        }
        self.coherence_info_text = ps.CornerText(self.coherence_plot, "bottom-right")
        x_range, y_range = (np.log10(0.1), np.log10(20)), (0, 1)
        self.coherence_plot.getViewBox().setDefaultRange(x_range, y_range)
        # Zooming out further would only show empty space around the plot
        self.coherence_plot.setLimits(
            xMin=x_range[0], xMax=x_range[1], yMin=y_range[0], yMax=y_range[1]
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

    def selectedRange(self, kind):
        """(t_start, t_stop) of the range, the estimation one for a validation
        range not selected: the data the model is then judged on."""
        if kind == kValidation and self.validation_range is not None:
            return self.validation_range
        return (self.t_start, self.t_stop)

    def removeRange(self, kind):
        """Without a validation range the model is judged on the estimation
        one, without an estimation range it is fitted to the whole log."""
        if kind == kValidation:
            self.validation_range = None
        else:
            (self.t_start, self.t_stop) = (None, None)
        for item in (*self.selection_regions[kind], *self.range_remove_buttons[kind]):
            item.hide()
        self.plotCoherence()

    def setSelectionRegion(self, kind, t_start, t_stop):
        for region, remove_button in zip(
            self.selection_regions[kind], self.range_remove_buttons[kind]
        ):
            region.blockSignals(True)
            region.setRegion((t_start, t_stop))
            region.blockSignals(False)
            region.show()
            self.placeRemoveButton(remove_button, region)
            remove_button.show()

    @staticmethod
    def placeRemoveButton(remove_button, region):
        view_top = region.getViewBox().viewRange()[1][1]
        remove_button.setPos(region.getRegion()[1], view_top)

    def onSpanDragged(self, t_start, t_stop, finished, button):
        kind = kValidation if button == Qt.RightButton else kEstimation
        self.setSelectionRegion(kind, t_start, t_stop)
        if finished:
            self.onselect(kind, t_start, t_stop)

    def onRegionChanged(self, kind, changed_region):
        t_start, t_stop = changed_region.getRegion()
        for region in self.selection_regions[kind]:
            if region is not changed_region:
                region.blockSignals(True)
                region.setRegion((t_start, t_stop))
                region.blockSignals(False)

    def onselect(self, kind, xmin, xmax):
        indmin, indmax = np.searchsorted(self.t, (xmin, xmax))
        indmax = min(len(self.t) - 1, indmax)
        indmin = min(indmin, indmax)

        t_range = (self.t[indmin], self.t[indmax])
        if kind == kValidation:
            self.validation_range = t_range
        else:
            (self.t_start, self.t_stop) = t_range
        self.setSelectionRegion(kind, *t_range)
        self.zoomOnSelectedRanges()
        self.plotCoherence()

    def zoomOnSelectedRanges(self):
        bounds = [
            t
            for kind in (kEstimation, kValidation)
            for t in self.selectedRange(kind)
            if t is not None
        ]
        if bounds:
            self.input_plot.setXRange(min(bounds) - 1.0, max(bounds) + 1.0, padding=0)

    def plotPIDAnalysis(self):
        if len(self.t) == 0 or len(self.u) == 0 or len(self.y) == 0:
            return

        t_sel, u_sel, y_sel, _ = self.getRangeData(kEstimation)

        # Create analysis window if not open yet
        if not hasattr(self, "pid_window") or self.pid_window is None:
            self.pid_window = PIDAnalyseWindow(self)

        # Show
        self.pid_window.show()
        self.pid_window.raise_()

        # Update plot
        self.pid_window.generate_step_response(u_sel, y_sel, t_sel)

    def getRangeData(self, kind):
        """Input, output and airspeed of the selected range, of the whole log
        when none is selected."""
        t_start, t_stop = self.selectedRange(kind)
        if t_start is not None and t_stop is not None and t_stop > t_start:
            return self.getInputOutputData(t_start, t_stop)
        return self.getInputOutputData()

    def plotCoherence(self):
        if len(self.t) == 0 or len(self.u) == 0 or len(self.y) == 0:
            return

        warnings = []
        info_lines = []
        for kind, curve in self.coherence_curves.items():
            curve.setData([], [])
            ps.setLegendEntry(self.coherence_legend, curve, None)
            if kind == kValidation and self.validation_range is None:
                continue
            name = f"{kind.capitalize()} range"
            freq, coherence, text = self.rangeCoherence(kind)
            if freq is None:
                warnings.append(f"{name}: {text}")
            else:
                curve.setData(freq, coherence)
                ps.setLegendEntry(self.coherence_legend, curve, name)
                info_lines.append((f"{name}: {text}", kRangeColors[kind]))

        self.coherence_info_text.setLines(info_lines)
        self.label_warning.setText("\n".join(warnings))
        self.label_warning.setVisible(bool(warnings))

    def rangeCoherence(self, kind):
        """(freq, coherence, info) of the range, (None, None, warning) when
        it cannot give one."""
        t_sel, u_sel, y_sel, _ = self.getRangeData(kind)
        num_samples = len(t_sel)
        duration = t_sel[-1] - t_sel[0]

        if hasMissingSamples(u_sel, y_sel):
            return None, None, kMissingSamplesWarning
        if num_samples < kMinCoherenceSamples:
            return (
                None,
                None,
                f"Increase the window size to at least {kMinCoherenceSamples} "
                f"samples. Currently selected: {duration:.2f} seconds, "
                f"{num_samples} samples.",
            )

        fs = 1 / np.mean(np.diff(t_sel))
        nperseg = min(1024, num_samples // 4)
        freq, coherence = signal.coherence(u_sel, y_sel, fs, nperseg=nperseg)
        info = (
            f"Samples: {num_samples}, Duration: {duration:.2f}s, "
            f"fs: {fs:.1f}Hz, nperseg: {nperseg}, Δf: {fs / nperseg:.2f}Hz"
        )
        return freq, coherence, info
