"""Functional tests for the filter chain widget (table + plot).

Driven headlessly (no real window). The add/edit dialog is stubbed so the
tests never open a modal. Run with:
    QT_QPA_PLATFORM=offscreen pytest test_filter_chain_widget.py
The offscreen platform is also set automatically below as a fallback.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import filter_chain_widget  # noqa: E402
import pytest  # noqa: E402
from filter_chain_widget import (  # noqa: E402
    COL_EDIT,
    COL_SUMMARY,
    COMBINED_COLOR,
    FilterChainWidget,
)
from filter_library import Filter, FilterChain  # noqa: E402
from PyQt5.QtWidgets import QApplication, QDialog  # noqa: E402

FS = 800.0


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def stub_dialog(monkeypatch):
    """Replace FilterEditDialog with one that returns a preset filter.

    Usage: call ``stub_dialog(some_filter)`` to make the next Add/Edit accept
    with that filter; pass ``None`` to simulate the user cancelling.
    """

    def _install(result_filter, accepted=True):
        class FakeDialog:
            Accepted = QDialog.Accepted

            def __init__(self, *a, **k):
                self.result_filter = result_filter

            def exec_(self):
                return QDialog.Accepted if accepted else QDialog.Rejected

        monkeypatch.setattr(filter_chain_widget, "FilterEditDialog", FakeDialog)

    return _install


def _line_colors(widget):
    """Map trace label -> color for the magnitude axis."""
    return {
        line.get_label(): line.get_color()
        for line in widget.canvas.ax_mag.get_lines()
        if line.get_label() and not line.get_label().startswith("_")
    }


def _make(*filters):
    return FilterChainWidget(fs=FS, chain=FilterChain(list(filters)))


# --- table content ----------------------------------------------------------
def test_rows_show_two_line_summary(qapp):
    w = _make(Filter("lpf2_damped", {"fc": 20.0, "zeta": 0.7}))
    text = w.table.item(0, COL_SUMMARY).text()
    assert "\n" in text
    name, params = text.split("\n", 1)
    assert name == "LPF 2nd order (damped)"
    assert params == "f_c: 20 Hz, Damping: 0.7"


def test_icon_button_labels(qapp):
    w = _make(Filter("lpf1_butter", {"fc": 20.0}))
    assert w.table.cellWidget(0, COL_EDIT).text() == filter_chain_widget.EDIT_LABEL


def test_table_fits_content(qapp):
    w = _make(Filter("lpf1_butter"), Filter("notch2"))
    header = w.table.horizontalHeader().height()
    rows = sum(w.table.rowHeight(r) for r in range(w.table.rowCount()))
    # Height hugs header + rows (allow for the frame border).
    assert abs(w.table.height() - (header + rows)) <= 4


# --- add / edit / remove ----------------------------------------------------
def test_add_filter_via_dialog(qapp, stub_dialog):
    w = _make()
    stub_dialog(Filter("lpf2_butter", {"fc": 30.0}))
    w._on_add()
    assert len(w.chain) == 1
    assert w.chain[0].type_id == "lpf2_butter"
    assert w.table.rowCount() == 1


def test_add_cancelled_changes_nothing(qapp, stub_dialog):
    w = _make(Filter("lpf1_butter"))
    stub_dialog(None, accepted=False)
    w._on_add()
    assert len(w.chain) == 1


def test_edit_filter_via_dialog(qapp, stub_dialog):
    w = _make(Filter("lpf1_butter", {"fc": 10.0}))
    stub_dialog(Filter("notch2", {"fc": 50.0, "bw": 5.0}))
    w._on_edit(0)
    assert w.chain[0].type_id == "notch2"
    assert "Notch" in w.table.item(0, COL_SUMMARY).text()


def test_remove_filter(qapp):
    w = _make(Filter("lpf1_butter"), Filter("notch2"))
    w._on_remove(0)
    assert len(w.chain) == 1
    assert w.chain[0].type_id == "notch2"
    assert w.table.rowCount() == 1
    assert len(w._enabled) == 1


# --- plotting ---------------------------------------------------------------
def test_combined_trace_always_present_and_black(qapp):
    w = _make(Filter("lpf2_butter", {"fc": 20.0}))
    colors = _line_colors(w)
    assert colors.get("Combined") == COMBINED_COLOR


def test_disable_removes_filter_from_graphs(qapp):
    w = _make(Filter("lpf2_butter", {"fc": 20.0}))
    # Enabled by default: combined + the one filter.
    assert "Combined" in _line_colors(w)
    assert len(_line_colors(w)) == 2

    w._on_enabled_toggled(0, False)
    # Disabling the only filter leaves no combined trace and no overlay.
    assert len(_line_colors(w)) == 0


def test_disable_excludes_filter_from_combined(qapp):
    lpf = Filter("lpf2_butter", {"fc": 20.0})
    notch = Filter("notch2", {"fc": 80.0, "bw": 30.0})
    w = _make(lpf, notch)

    w._on_enabled_toggled(1, False)  # disable the notch

    # The combined chain now equals just the enabled (low-pass) filter.
    b, a = w.enabled_chain().coefficients(w.fs)
    b_lpf, a_lpf = FilterChain([lpf]).coefficients(w.fs)
    assert list(b) == list(b_lpf)
    assert list(a) == list(a_lpf)


def test_rows_enabled_by_default(qapp):
    w = _make(Filter("lpf1_butter"), Filter("notch2"))
    assert w._enabled == [True, True]


def test_colors_stable_across_enable_toggles(qapp):
    w = _make(
        Filter("lpf2_butter", {"fc": 20.0}),
        Filter("notch2", {"fc": 80.0, "bw": 30.0}),
        Filter("hpf1_butter", {"fc": 5.0}),
    )
    all_shown = _line_colors(w)

    w._on_enabled_toggled(1, False)  # disable the middle filter
    reduced = _line_colors(w)

    common = set(all_shown) & set(reduced)
    assert common  # sanity
    assert all(all_shown[label] == reduced[label] for label in common)


def test_fs_change_triggers_signal(qapp):
    w = _make(Filter("lpf1_butter"))
    fired = []
    w.changed.connect(lambda: fired.append(True))
    w.spin_fs.setValue(1000.0)
    assert fired
