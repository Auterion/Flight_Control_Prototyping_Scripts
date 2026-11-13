"""Tests of the sessions: saving and loading the state of the interface by group.

Run with:  QT_QPA_PLATFORM=offscreen poetry run pytest test_session.py
"""

import os
import shutil

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import presets  # noqa: E402
import pytest  # noqa: E402
import session as session_module  # noqa: E402
import yaml  # noqa: E402
from data_selection_window import DataSelectionWindow  # noqa: E402
from PyQt5.QtCore import Qt  # noqa: E402
from PyQt5.QtWidgets import QApplication, QMessageBox  # noqa: E402
from session import (  # noqa: E402
    SessionError,
    SessionGroupsDialog,
    isSameLog,
    kSessionGroups,
    kSessionVersion,
    loadSession,
    resolveLogPath,
    saveSession,
)

import autotune  # noqa: E402

LOG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "logs", "quadrotor_x500.ulg"
)
DT = 0.005
NUM = [0.0, 0.05, 0.03]
DEN = [1.0, -1.6, 0.65]
GAINS = {"P": 0.21, "I": 1.7, "D": 0.013, "FF": 0.05}


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def selection(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(
        presets, "presets_file_path", lambda: str(tmp_path / "presets.yaml")
    )
    window = DataSelectionWindow(LOG)
    window.combo_preset.setCurrentIndex(window.combo_preset.findText("Rollrate"))
    window.t_start, window.t_stop = 76.5, 83.5
    return window


def load_log(selection, monkeypatch):
    selection.exec_ = lambda: selection.loadSelection() or True
    monkeypatch.setattr(autotune, "DataSelectionWindow", lambda *args: selection)
    window = autotune.Window()
    window.loadLog()
    monkeypatch.setattr(autotune, "DataSelectionWindow", DataSelectionWindow)
    return window


def save_and_load(window, path, groups=kSessionGroups, into=None):
    """Session of window saved to path and loaded into a new window."""
    states = window.sessionState()
    saveSession(path, {group: states[group] for group in groups})
    session = loadSession(path)
    if "data" in session:
        session["data"]["log"] = resolveLogPath(path, session["data"]["log"])
    loaded = into or autotune.Window()
    loaded.applySession(session, list(groups))
    return loaded


def model_session(num=NUM, den=DEN, delays=2, dt=DT):
    return {"model": {"dt": dt, "delays": delays, "num": num, "den": den}}


def tuning_session(window, **changes):
    tuning = window.tuningState()
    tuning.update(changes)
    return {"tuning": tuning}


def curve_data(curve):
    return np.asarray(curve.xData), np.asarray(curve.yData)


def test_session_reproduces_every_group_exactly(selection, monkeypatch, tmp_path):
    selection.validation_range = (103.0, 109.0)
    window = load_log(selection, monkeypatch)
    window.remove_offset_box.setChecked(False)
    window.f_hp_spinbox.setValue(0.5)
    window.f_lp_spinbox.setValue(25.0)
    window.line_edit_poles.setValue(3)
    window.line_edit_delays.setValue(2)
    window.horizon_spinbox.setValue(0.05)
    window.onSysIdClicked()
    window.t_coeffs.item(0, 0).setText("-1.234567")
    window.updateModel()
    window.slider_rise_time.setValue(0.2)
    window.slider_detune.setValue(0.8)
    window.gain_form_radio["parallel"].setChecked(True)
    window.gains = dict(GAINS)
    window.negate_control_box.setChecked(True)
    window.updateKIDSliders()
    window.updateClosedLoop()
    window.step_sim_spinbox["step_duration"].setValue(3.0)
    window.step_info_spinbox["overshoot"].setValue(15.0)
    window.line_edit_offset.setValue(0.1)

    loaded = save_and_load(window, str(tmp_path / "session.yaml"))

    assert loaded.sessionState() == window.sessionState()
    assert loaded.num == window.num
    assert loaded.den == window.den
    assert loaded.den[1] == -1.234567
    assert loaded.lbl_fit.text() == window.lbl_fit.text()
    assert loaded.lbl_closed_loop_fit.text() == window.lbl_closed_loop_fit.text()
    for curve in ("step_response_curve", "model_curve", "input_curve"):
        x, y = curve_data(getattr(loaded, curve))
        x_saved, y_saved = curve_data(getattr(window, curve))
        assert np.array_equal(x, x_saved) and np.array_equal(y, y_saved)
    assert loaded.gain_edit["parallel"]["P"].text() == "0.2100"
    assert loaded.lbl_rise_time.text() == "0.20"


def test_session_file_is_readable_yaml(selection, monkeypatch, tmp_path):
    window = load_log(selection, monkeypatch)
    path = tmp_path / "session.yaml"
    saveSession(str(path), {"data": window.dataState(), **model_session()})

    session = yaml.safe_load(path.read_text())

    assert session["format"] == "px4-autotune-session"
    assert session["version"] == kSessionVersion
    assert session["data"]["preset"]["name"] == "Rollrate"
    assert session["data"]["input"] == selection.combo_u.currentText()
    assert session["data"]["estimation_range"] == [76.5, 83.5]
    assert session["data"]["validation_range"] is None
    assert session["data"]["log"]["absolute"] == LOG
    assert session["model"]["num"] == NUM
    assert "preprocessing" not in session


def test_model_loads_without_data(qapp):
    window = autotune.Window()

    window.applySession(model_session(), ["model"])

    assert window.is_system_identified
    assert window.num == NUM and window.den == DEN
    assert window.sys_id_delays == 2
    assert window.line_edit_poles.value() == 2
    assert len(window.step_response_curve.yData) > 0
    assert len(window.bode_mag_curves["plant"].yData) > 0
    assert window.lbl_fit.text() == "—"
    assert window.lbl_stability.text() == "✓ Stable"
    assert not window.btn_run_sys_id.isEnabled()


def test_tuning_keeps_its_gains_on_another_model(selection, monkeypatch):
    preset = selection.preset_candidates["Rollrate"]
    preset["pi_no_zero"] = True
    window = load_log(selection, monkeypatch)
    tuning = tuning_session(
        window, controller_tab="GMVC", gains=dict(GAINS), pi_no_zero=False
    )

    window.applySession(tuning, ["tuning"])

    assert window.isGmvcTabActive()
    assert window.gains == GAINS
    assert not window.pid_no_zero_box.isChecked()


def test_tuning_overrides_the_flown_controller_of_loaded_data(
    selection, monkeypatch, tmp_path
):
    window = load_log(selection, monkeypatch)
    window.gains = dict(GAINS)

    loaded = save_and_load(window, str(tmp_path / "s.yaml"), ["data", "tuning"])

    assert loaded.gains == GAINS


def test_data_without_model_is_identified(selection, monkeypatch, tmp_path):
    window = load_log(selection, monkeypatch)

    loaded = save_and_load(window, str(tmp_path / "s.yaml"), ["data"])

    assert loaded.num == window.num
    assert loaded.gains == window.gains
    assert not loaded.btn_run_sys_id.isEnabled()
    assert loaded.btn_find_params.isEnabled()


def test_model_without_its_data_asks_for_an_identification(selection, monkeypatch):
    window = load_log(selection, monkeypatch)

    window.applySession(model_session(dt=window.dt), ["model"])

    assert window.num == NUM
    assert window.btn_run_sys_id.isEnabled()


def test_model_sampled_differently_resamples_the_data(selection, monkeypatch):
    window = load_log(selection, monkeypatch)

    window.applySession(model_session(dt=0.01), ["model"])

    assert window.dt == 0.01
    assert np.mean(np.diff(window.t)) == pytest.approx(0.01)


def test_missing_keys_keep_the_current_values(qapp):
    window = autotune.Window()
    step_info = dict(window.step_info)

    window.applySession({"others": {"model_output_offset": 0.3}}, ["others"])

    assert window.line_edit_offset.value() == 0.3
    assert window.step_info == step_info


def test_groups_not_chosen_are_not_loaded(qapp):
    window = autotune.Window()

    loaded = window.applySession(model_session(), ["tuning"])

    assert loaded == []
    assert not window.is_system_identified


def test_running_parameter_search_does_not_override_the_loaded_model(
    selection, monkeypatch
):
    window = load_log(selection, monkeypatch)
    window.onFindParamsClicked()

    window.applySession(model_session(dt=window.dt), ["model"])
    QApplication.processEvents()

    assert window.num == NUM
    assert window.btn_find_params.text() == "Find parameters"


def test_select_data_shows_the_current_selection(selection, monkeypatch):
    window = load_log(selection, monkeypatch)
    opened = []
    selection.exec_ = lambda: False
    monkeypatch.setattr(
        autotune,
        "DataSelectionWindow",
        lambda *args: opened.append(args) or selection,
    )

    window.loadLog()

    assert opened == [(LOG, window.data_selection)]
    assert window.data_selection["estimation_range"] == [76.5, 83.5]


def test_session_finds_its_log_moved_with_it(tmp_path):
    folder = tmp_path / "flight"
    folder.mkdir()
    (folder / "log.ulg").write_bytes(b"ulog")
    saveSession(
        str(folder / "session.yaml"), {"data": {"log": str(folder / "log.ulg")}}
    )
    shutil.move(folder, tmp_path / "moved")
    session = loadSession(str(tmp_path / "moved" / "session.yaml"))

    log_path = resolveLogPath(
        str(tmp_path / "moved" / "session.yaml"), session["data"]["log"]
    )

    assert session["data"]["log"]["path"] == "log.ulg"
    assert log_path == str(tmp_path / "moved" / "log.ulg")
    assert isSameLog(log_path, session["data"]["log"])
    (tmp_path / "moved" / "log.ulg").write_bytes(b"other ulog")
    assert not isSameLog(log_path, session["data"]["log"])


def test_missing_log_is_not_found(tmp_path):
    entry = {"path": "log.ulg", "absolute": str(tmp_path / "nowhere" / "log.ulg")}

    assert resolveLogPath(str(tmp_path / "session.yaml"), entry) is None


def test_different_log_asks_before_loading(qapp, tmp_path, monkeypatch):
    (tmp_path / "log.ulg").write_bytes(b"ulog")
    entry = {"path": "log.ulg", "sha256": "0" * 64}
    asked = []
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args: asked.append(args) or QMessageBox.No
    )
    window = autotune.Window()

    log_path = window.locateSessionLog(str(tmp_path / "session.yaml"), entry)

    assert log_path is None
    assert "differs" in asked[0][2]


def test_newer_session_version_is_refused(tmp_path):
    path = tmp_path / "session.yaml"
    path.write_text(yaml.safe_dump({"format": "px4-autotune-session", "version": 99}))

    with pytest.raises(SessionError, match="newer version"):
        loadSession(str(path))


def test_other_yaml_is_not_a_session(tmp_path):
    path = tmp_path / "presets.yaml"
    path.write_text(yaml.safe_dump({"Rollrate": {"input": "a", "output": "b"}}))

    with pytest.raises(SessionError, match="not an autotune session"):
        loadSession(str(path))


def test_fresh_window_has_no_data_or_model_to_save(qapp):
    window = autotune.Window()

    states = window.sessionState()
    dialog = SessionGroupsDialog(window, "Save session", states, "Nothing to save")

    assert states["data"] is None and states["model"] is None
    assert not dialog.group_boxes["data"].isEnabled()
    assert not dialog.group_boxes["model"].isChecked()
    assert dialog.selectedGroups() == [
        "preprocessing",
        "model_structure",
        "tuning",
        "others",
    ]


def test_dialog_without_a_chosen_group_cannot_be_accepted(qapp):
    dialog = SessionGroupsDialog(None, "Load session", model_session(), "Not in it")

    dialog.group_boxes["model"].setChecked(False)

    assert not dialog.buttons.button(dialog.buttons.Ok).isEnabled()


def test_buttons_save_and_load_the_chosen_groups(selection, monkeypatch, tmp_path):
    window = load_log(selection, monkeypatch)
    window.gains = dict(GAINS)
    path = str(tmp_path / "flight")
    monkeypatch.setattr(SessionGroupsDialog, "exec_", lambda dialog: True)
    monkeypatch.setattr(
        autotune.QFileDialog, "getSaveFileName", lambda *args, **kwargs: (path, "")
    )
    monkeypatch.setattr(
        autotune.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (path + ".yaml", ""),
    )

    window.btn_save_session.click()
    loaded = autotune.Window()
    loaded.btn_load_session.click()

    assert set(loadSession(path + ".yaml")) >= set(kSessionGroups)
    assert loaded.sessionState() == window.sessionState()


def test_log_in_a_sibling_folder_is_saved_relative(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "log.ulg").write_bytes(b"ulog")

    entry = session_module.logEntry(
        str(tmp_path / "logs" / "log.ulg"), str(tmp_path / "sessions" / "s.yaml")
    )

    assert entry["path"] == os.path.join("..", "logs", "log.ulg")
    assert entry["absolute"] == str(tmp_path / "logs" / "log.ulg")


def test_all_box_checks_and_unchecks_every_choosable_group(qapp):
    states = {**model_session(), "others": {}}
    dialog = SessionGroupsDialog(None, "Load session", states, "Not in it")
    dialog.group_boxes["others"].setChecked(False)

    assert dialog.all_box.checkState() == Qt.PartiallyChecked
    dialog.all_box.click()
    assert dialog.selectedGroups() == ["model", "others"]
    assert dialog.all_box.checkState() == Qt.Checked
    dialog.all_box.click()
    assert dialog.selectedGroups() == []
    assert dialog.all_box.checkState() == Qt.Unchecked
    assert not dialog.buttons.button(dialog.buttons.Ok).isEnabled()
    dialog.all_box.click()
    assert dialog.selectedGroups() == ["model", "others"]


def test_all_box_follows_the_groups(qapp):
    dialog = SessionGroupsDialog(None, "Load session", model_session(), "Not in it")

    assert dialog.all_box.checkState() == Qt.Checked
    dialog.group_boxes["model"].setChecked(False)
    assert dialog.all_box.checkState() == Qt.Unchecked


def test_loading_data_with_a_tuning_does_not_ask_for_the_flown_gains(
    selection, monkeypatch, tmp_path
):
    window = load_log(selection, monkeypatch)
    window.gains = dict(GAINS)
    loaded = load_log(selection, monkeypatch)
    loaded.gains = {"P": 0.5, "I": 1.0, "D": 0.01, "FF": 0.2}
    monkeypatch.setattr(
        autotune.QMessageBox, "question", lambda *args: pytest.fail("asked")
    )

    save_and_load(window, str(tmp_path / "s.yaml"), ["data", "tuning"], into=loaded)

    assert loaded.gains == GAINS


def test_restart_restores_the_session(selection, monkeypatch):
    window = load_log(selection, monkeypatch)
    window.line_edit_poles.setValue(3)
    window.onSysIdClicked()
    window.gains = dict(GAINS)
    window.updateKIDSliders()
    window.updateClosedLoop()
    restart_args = []
    monkeypatch.setattr(autotune, "restart", lambda *args: restart_args.extend(args))

    window.update_banner.btn_restart.click()
    option, path = restart_args
    restarted = autotune.Window()
    restarted.restoreSession(path)

    assert option == "--restore-session"
    assert restarted.sessionState() == window.sessionState()
    assert not os.path.exists(path)
