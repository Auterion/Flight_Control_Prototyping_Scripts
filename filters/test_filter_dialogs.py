"""Functional tests for the add/edit filter dialog.

These drive FilterEditDialog headlessly (no real window is shown). Run with:
    QT_QPA_PLATFORM=offscreen pytest test_filter_dialogs.py
The offscreen platform is also set automatically below as a fallback.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from filter_edit_dialog import FilterEditDialog  # noqa: E402
from filter_library import Filter  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

FS = 800.0


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _param_keys(dialog):
    return set(dialog._param_widgets.keys())


def test_add_mode_defaults(qapp):
    d = FilterEditDialog(fs=FS)
    assert d.windowTitle() == "Add filter"
    # First registry entry is preselected and its params are shown.
    assert d.combo_type.currentData() == "lpf1_butter"
    assert _param_keys(d) == {"fc"}
    assert d.result_filter is None


def test_type_switch_rebuilds_params(qapp):
    d = FilterEditDialog(fs=FS)
    d.combo_type.setCurrentIndex(d.combo_type.findData("notch2"))
    assert _param_keys(d) == {"fc", "bw"}
    d.combo_type.setCurrentIndex(d.combo_type.findData("lpf2_damped"))
    assert _param_keys(d) == {"fc", "zeta"}


def test_edit_mode_prefills(qapp):
    flt = Filter("lpf2_damped", {"fc": 15.0, "zeta": 0.7})
    d = FilterEditDialog(fs=FS, flt=flt)
    assert d.windowTitle() == "Edit filter"
    assert d.combo_type.currentData() == "lpf2_damped"
    assert d._param_widgets["fc"].value() == pytest.approx(15.0)
    assert d._param_widgets["zeta"].value() == pytest.approx(0.7)


def test_accept_builds_filter_from_widgets(qapp):
    d = FilterEditDialog(fs=FS)
    d.combo_type.setCurrentIndex(d.combo_type.findData("notch2"))
    d._param_widgets["fc"].setValue(120.0)
    d._param_widgets["bw"].setValue(10.0)

    d._on_accept()

    assert d.result() == FilterEditDialog.Accepted
    assert d.result_filter.type_id == "notch2"
    assert d.result_filter.params == {"fc": 120.0, "bw": 10.0}


def test_edit_then_change_params(qapp):
    d = FilterEditDialog(fs=FS, flt=Filter("lpf2_butter", {"fc": 20.0}))
    d._param_widgets["fc"].setValue(50.0)
    d._on_accept()
    assert d.result_filter.params == {"fc": 50.0}
