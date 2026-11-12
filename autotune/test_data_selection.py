"""Tests of the time range selection and coherence plot of the data selection.

Run with:  QT_QPA_PLATFORM=offscreen poetry run pytest test_data_selection.py
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import plot_style as ps  # noqa: E402
import presets  # noqa: E402
import pytest  # noqa: E402
from data_extractor import resample_interp  # noqa: E402
from data_selection_window import (  # noqa: E402
    DataSelectionWindow,
    kEstimation,
    kRangeColors,
    kValidation,
)
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


def drag_span(selection, t_start, t_stop, button=Qt.LeftButton):
    view_box = selection.input_plot.getViewBox()
    view_box.sigSpanDragged.emit(t_start, t_stop, False, int(button))
    view_box.sigSpanDragged.emit(t_start, t_stop, True, int(button))


def is_empty(curve):
    return curve.xData is None or len(curve.xData) == 0


def legend_names(selection):
    return [label.text for _, label in selection.coherence_legend.items]


def regions(selection, kind=kEstimation):
    return [region.getRegion() for region in selection.selection_regions[kind]]


def drag_validation_span(selection, t_start, t_stop):
    drag_span(selection, t_start, t_stop, Qt.RightButton)


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

    freq, coherence = selection.coherence_curves[kEstimation].getData()
    assert len(freq) > 0
    assert np.all((coherence >= 0) & (coherence <= 1))
    assert selection.coherence_info_text.text().startswith("Estimation range: Samples:")
    assert selection.label_warning.isHidden()


def test_coherence_plot_marks_the_minimum_good_coherence(selection):
    line = selection.coherence_threshold_line

    assert line in selection.coherence_plot.items
    assert line.value() == pytest.approx(0.6)


def test_selection_of_a_couple_of_seconds_plots_the_coherence(selection):
    drag_span(selection, 76.5, 78.0)

    assert len(selection.coherence_curves[kEstimation].getData()[0]) > 0
    assert selection.label_warning.isHidden()


def test_short_selection_warns_and_clears_the_coherence(selection):
    drag_span(selection, 76.5, 83.5)
    drag_span(selection, 76.5, 77.0)

    assert not selection.label_warning.isHidden()
    assert is_empty(selection.coherence_curves[kEstimation])
    assert selection.coherence_info_text.text() == ""


def test_moving_a_region_moves_the_other_and_selects(selection):
    drag_span(selection, 76.5, 83.5)
    output_region = selection.selection_regions[kEstimation][1]

    output_region.setRegion((70.0, 80.0))

    assert selection.t_start == pytest.approx(70.0, abs=0.1)
    assert selection.t_stop == pytest.approx(80.0, abs=0.1)
    for region in regions(selection):
        assert region == pytest.approx((selection.t_start, selection.t_stop))


def test_validation_is_the_estimation_range_until_selected(selection):
    drag_span(selection, 76.5, 83.5)

    assert selection.validation_range is None
    assert selection.selectedRange(kValidation) == (
        selection.t_start,
        selection.t_stop,
    )
    assert not any(
        region.isVisible() for region in selection.selection_regions[kValidation]
    )


def test_validation_drag_selects_its_own_range(selection):
    drag_span(selection, 76.5, 83.5)
    estimation = (selection.t_start, selection.t_stop)

    drag_validation_span(selection, 103.0, 109.0)

    assert (selection.t_start, selection.t_stop) == estimation
    assert selection.validation_range[0] == pytest.approx(103.0, abs=0.1)
    assert selection.validation_range[1] == pytest.approx(109.0, abs=0.1)
    assert selection.validation_range[0] in selection.t
    for region in regions(selection, kValidation):
        assert region == pytest.approx(selection.validation_range)
    for region in regions(selection):
        assert region == pytest.approx(estimation)


def test_view_zooms_on_both_ranges(selection):
    drag_span(selection, 76.5, 83.5)
    drag_validation_span(selection, 103.0, 109.0)

    x_min, x_max = selection.input_plot.getViewBox().viewRange()[0]
    assert x_min == pytest.approx(selection.t_start - 1.0)
    assert x_max == pytest.approx(selection.validation_range[1] + 1.0)


def test_coherences_of_both_ranges_are_plotted_in_their_colors(selection):
    drag_span(selection, 76.5, 83.5)
    estimation_coherence = selection.coherence_curves[kEstimation].yData.copy()
    assert is_empty(selection.coherence_curves[kValidation])
    assert legend_names(selection) == ["Estimation range"]

    drag_validation_span(selection, 103.0, 109.0)

    validation_coherence = selection.coherence_curves[kValidation].yData
    assert selection.coherence_curves[kEstimation].yData == pytest.approx(
        estimation_coherence
    )
    assert not np.array_equal(validation_coherence, estimation_coherence)
    for kind, curve in selection.coherence_curves.items():
        assert curve.opts["pen"].color() == ps.color(kRangeColors[kind])
    assert legend_names(selection) == ["Estimation range", "Validation range"]
    lines = selection.coherence_info_text.text().splitlines()
    assert lines[0].startswith("Estimation range:")
    assert lines[1].startswith("Validation range:")


def test_short_validation_range_warns_and_keeps_the_estimation_coherence(selection):
    drag_span(selection, 76.5, 83.5)

    drag_validation_span(selection, 103.0, 103.5)

    assert not is_empty(selection.coherence_curves[kEstimation])
    assert is_empty(selection.coherence_curves[kValidation])
    assert selection.label_warning.text().startswith("Validation range:")
    assert not selection.label_warning.isHidden()


def test_coherence_cannot_zoom_out_past_its_default_range(selection):
    view_box = selection.coherence_plot.getViewBox()

    view_box.scaleBy(x=10.0)

    (x_min, x_max), (y_min, y_max) = view_box.viewRange()
    assert x_min >= np.log10(0.1) - 1e-9
    assert x_max <= np.log10(20) + 1e-9
    assert (y_min, y_max) == pytest.approx((0.0, 1.0))


def test_moving_a_validation_region_keeps_the_estimation_range(selection):
    drag_span(selection, 76.5, 83.5)
    drag_validation_span(selection, 103.0, 109.0)

    selection.selection_regions[kValidation][1].setRegion((110.0, 120.0))

    assert selection.validation_range[0] == pytest.approx(110.0, abs=0.1)
    assert selection.t_start == pytest.approx(76.5, abs=0.1)
    for region in regions(selection, kValidation):
        assert region == pytest.approx(selection.validation_range)


class FakeClickEvent:
    def __init__(self, button=Qt.LeftButton):
        self.pressed = button
        self.accepted = False

    def button(self):
        return self.pressed

    def accept(self):
        self.accepted = True


def test_each_range_has_a_remove_button_at_its_top_right_corner(selection):
    drag_span(selection, 76.5, 83.5)
    drag_validation_span(selection, 103.0, 109.0)
    selection.selection_regions[kValidation][0].setRegion((110.0, 120.0))

    for kind, buttons in selection.range_remove_buttons.items():
        for button, region in zip(buttons, selection.selection_regions[kind]):
            assert button.isVisible()
            view_top = region.getViewBox().viewRange()[1][1]
            assert button.pos().x() == pytest.approx(region.getRegion()[1])
            assert button.pos().y() == pytest.approx(view_top)


def test_remove_button_stays_at_the_top_when_the_view_moves(selection):
    drag_span(selection, 76.5, 83.5)
    view_box = selection.input_plot.getViewBox()

    view_box.setYRange(-5.0, 7.0, padding=0)

    assert selection.range_remove_buttons[kEstimation][0].pos().y() == pytest.approx(
        7.0
    )


def test_removing_the_validation_range_validates_on_the_estimation_range(selection):
    drag_span(selection, 76.5, 83.5)
    drag_validation_span(selection, 103.0, 109.0)

    selection.range_remove_buttons[kValidation][0].mouseClickEvent(FakeClickEvent())

    assert selection.validation_range is None
    assert selection.selectedRange(kValidation) == (
        selection.t_start,
        selection.t_stop,
    )
    for item in (
        *selection.selection_regions[kValidation],
        *selection.range_remove_buttons[kValidation],
    ):
        assert not item.isVisible()
    assert is_empty(selection.coherence_curves[kValidation])
    assert all(
        region.isVisible() for region in selection.selection_regions[kEstimation]
    )


def test_removing_the_estimation_range_selects_the_whole_log(selection):
    drag_span(selection, 76.5, 83.5)

    selection.range_remove_buttons[kEstimation][1].mouseClickEvent(FakeClickEvent())

    assert (selection.t_start, selection.t_stop) == (None, None)
    assert not any(
        region.isVisible() for region in selection.selection_regions[kEstimation]
    )
    t_whole = selection.getInputOutputData()[0]
    assert selection.coherence_info_text.text().startswith(
        f"Estimation range: Samples: {len(t_whole)},"
    )


def test_right_click_does_not_remove_a_range(selection):
    drag_span(selection, 76.5, 83.5)

    selection.range_remove_buttons[kEstimation][0].mouseClickEvent(
        FakeClickEvent(Qt.RightButton)
    )

    assert selection.t_start is not None


class FakeDragEvent:
    def __init__(self, view_box, t_down, t_now, finished, button=Qt.LeftButton):
        self.down = view_box.mapViewToScene(QPointF(t_down, 0))
        self.now = view_box.mapViewToScene(QPointF(t_now, 0))
        self.finished = finished
        self.pressed = button

    def button(self):
        return self.pressed

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


def test_resampling_keeps_the_gaps_of_missing_samples():
    t = np.arange(10.0)
    u = np.where((t >= 3) & (t <= 5), np.nan, t)

    resampled = resample_interp(t, u, np.arange(0.0, 9.5, 0.5))

    is_missing = np.isnan(resampled)
    assert np.array_equal(np.flatnonzero(is_missing), np.arange(5, 12))
    assert resampled[~is_missing] == pytest.approx(
        np.arange(0.0, 9.5, 0.5)[~is_missing]
    )


def test_right_drag_draws_the_validation_range(selection):
    selection.resize(1000, 800)
    selection.show()
    view_box = selection.output_plot.getViewBox()
    view_box.setXRange(60.0, 120.0, padding=0)
    view_box.mouseDragEvent(FakeDragEvent(view_box, 76.5, 83.5, finished=True))

    view_box.mouseDragEvent(
        FakeDragEvent(view_box, 109.0, 103.0, finished=True, button=Qt.RightButton)
    )

    assert selection.validation_range[0] == pytest.approx(103.0, abs=0.1)
    assert selection.validation_range[1] == pytest.approx(109.0, abs=0.1)
    assert selection.t_start == pytest.approx(76.5, abs=0.1)
    assert selection.range_remove_buttons[kValidation][1].isVisible()
