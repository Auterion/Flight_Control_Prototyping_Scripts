"""Tests of the time range selection and coherence plot of the data selection.

Run with:  QT_QPA_PLATFORM=offscreen poetry run pytest test_data_selection.py
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import presets  # noqa: E402
import pytest  # noqa: E402
from data_selection_window import DataSelectionWindow  # noqa: E402
from PyQt5.QtCore import QPointF, Qt  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def selection(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(
        presets, "presets_file_path", lambda: str(tmp_path / "presets.yaml")
    )
    log = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    window = DataSelectionWindow(os.path.join(log, "quadrotor_x500.ulg"))
    window.combo_preset.setCurrentIndex(window.combo_preset.findText("Rollrate"))
    return window


def drag_span(selection, t_start, t_stop):
    view_box = selection.input_plot.getViewBox()
    view_box.sigSpanDragged.emit(t_start, t_stop, False)
    view_box.sigSpanDragged.emit(t_start, t_stop, True)


def regions(selection):
    return [region.getRegion() for region in selection.selection_regions]


def test_logged_signals_are_plotted(selection):
    for curve, index in (
        (selection.input_curve, selection.index_u),
        (selection.output_curve, selection.index_y),
    ):
        t, values = selection.data_extractor.getPreview(selection.topics[index])
        assert curve.xData == pytest.approx(t)
        assert curve.yData == pytest.approx(values)


def test_dragging_selects_a_range_snapped_to_samples(selection):
    drag_span(selection, 76.5, 83.5)

    assert selection.t_start in selection.t
    assert selection.t_stop in selection.t
    assert selection.t_start == pytest.approx(76.5, abs=0.1)
    assert selection.t_stop == pytest.approx(83.5, abs=0.1)
    for region in regions(selection):
        assert region == pytest.approx((selection.t_start, selection.t_stop))


def test_selection_zooms_on_the_range(selection):
    drag_span(selection, 76.5, 83.5)

    x_min, x_max = selection.input_plot.getViewBox().viewRange()[0]
    assert x_min == pytest.approx(selection.t_start - 1.0)
    assert x_max == pytest.approx(selection.t_stop + 1.0)


def test_selection_plots_the_coherence(selection):
    drag_span(selection, 76.5, 83.5)

    freq, coherence = selection.coherence_curve.getData()
    assert len(freq) > 0
    assert np.all((coherence >= 0) & (coherence <= 1))
    assert selection.coherence_info_text.text().startswith("Samples:")
    assert selection.label_warning.isHidden()


def test_short_selection_warns_and_clears_the_coherence(selection):
    drag_span(selection, 76.5, 83.5)
    drag_span(selection, 76.5, 78.0)

    assert not selection.label_warning.isHidden()
    assert selection.coherence_curve.xData is None or (
        len(selection.coherence_curve.xData) == 0
    )
    assert selection.coherence_info_text.text() == ""


def test_moving_a_region_moves_the_other_and_selects(selection):
    drag_span(selection, 76.5, 83.5)
    output_region = selection.selection_regions[1]

    output_region.setRegion((70.0, 80.0))

    assert selection.t_start == pytest.approx(70.0, abs=0.1)
    assert selection.t_stop == pytest.approx(80.0, abs=0.1)
    for region in regions(selection):
        assert region == pytest.approx((selection.t_start, selection.t_stop))


class FakeDragEvent:
    def __init__(self, view_box, t_down, t_now, finished):
        self.down = view_box.mapViewToScene(QPointF(t_down, 0))
        self.now = view_box.mapViewToScene(QPointF(t_now, 0))
        self.finished = finished

    def button(self):
        return Qt.LeftButton

    def buttonDownScenePos(self):
        return self.down

    def scenePos(self):
        return self.now

    def isFinish(self):
        return self.finished

    def accept(self):
        pass


def test_left_drag_draws_the_selection_in_either_direction(selection):
    selection.resize(1000, 800)
    selection.show()
    view_box = selection.output_plot.getViewBox()
    view_box.setXRange(60.0, 100.0, padding=0)

    view_box.mouseDragEvent(FakeDragEvent(view_box, 85.0, 75.0, finished=True))

    assert selection.t_start == pytest.approx(75.0, abs=0.1)
    assert selection.t_stop == pytest.approx(85.0, abs=0.1)
