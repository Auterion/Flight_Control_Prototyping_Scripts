"""Functional tests for the preset add/edit/rename/delete dialog.

These drive PresetEditDialog headlessly (no real window is shown). Run with:
    QT_QPA_PLATFORM=offscreen poetry run pytest test_preset_dialogs.py
The offscreen platform is also set automatically below as a fallback.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import preset_dialogs  # noqa: E402
import pytest  # noqa: E402
from preset_dialogs import PresetEditDialog  # noqa: E402
from PyQt5.QtWidgets import QApplication, QMessageBox  # noqa: E402

EXISTING = {
    "Rollrate": {"input": "a/in.0", "output": "a/out.0", "input_legacy": "a/leg.0"},
    "Pitchrate": {"input": "b/in.0", "output": "b/out.0"},
}
TOPICS = ["a/in.0", "a/out.0", "b/in.0", "b/out.0", "c/in.0", "c/out.0", "a/leg.0"]
SEL_IN, SEL_OUT = "c/in.0", "c/out.0"  # "currently selected" main-window signals


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def make(qapp):
    def _make(original_name, create=False):
        # Fresh copy of EXISTING per dialog so tests stay isolated.
        existing = {k: dict(v) for k, v in EXISTING.items()}
        return PresetEditDialog(
            None, create, original_name, existing, TOPICS, SEL_IN, SEL_OUT
        )

    return _make


@pytest.fixture
def no_modals(monkeypatch):
    """Stub the modal warning/question boxes so headless tests never block."""
    monkeypatch.setattr(preset_dialogs.QMessageBox, "warning", lambda *a, **k: None)
    monkeypatch.setattr(
        preset_dialogs.QMessageBox, "question", lambda *a, **k: QMessageBox.Yes
    )


def test_edit_mode_defaults_to_current_selection(make):
    d = make("Rollrate")
    assert d._mode() == "update"
    assert d.btn_confirm.text() == "Update preset"
    # old column shows the stored preset, new combos default to the selection
    assert d.label_old_input.text() == "a/in.0"
    assert d.combo_new_input.currentText() == SEL_IN
    assert d.combo_new_output.currentText() == SEL_OUT
    assert d._delete_target() == "Rollrate"


def test_typing_unknown_name_renames(make):
    d = make("Rollrate")
    d.combo_name.setEditText("Rollrate_v2")
    assert d._mode() == "rename"
    assert d.btn_confirm.text() == "Update preset"

    d._on_confirm()

    assert d.result_action == "save"
    assert d.name == "Rollrate_v2"
    assert d.remove_name == "Rollrate"  # old key removed by caller
    # legacy fallback is carried over from the renamed preset
    assert d.preset == {
        "input": SEL_IN,
        "output": SEL_OUT,
        "input_legacy": "a/leg.0",
    }


def test_edit_button_opens_in_update_mode(make):
    d = make("Rollrate")  # Edit button -> create=False
    assert d._mode() == "update"
    assert d.windowTitle() == "Edit preset"


def test_add_button_creates(make):
    d = make(None, create=True)  # Add button -> create=True
    assert d._mode() == "create"
    assert d.windowTitle() == "Add preset"
    assert d.btn_confirm.text() == "Create new preset"
    assert d.combo_name.currentText() == ""
    assert d._delete_target() is None  # nothing to delete while creating

    d.combo_name.setEditText("BrandNew")
    d._on_confirm()

    assert d.result_action == "save"
    assert d.name == "BrandNew"
    assert d.remove_name is None  # nothing removed
    assert d.preset == {"input": SEL_IN, "output": SEL_OUT}


def test_renaming_to_existing_name_is_rejected(make, no_modals):
    # Editing "Rollrate" and changing its name to another existing preset must
    # be refused with feedback - never a silent overwrite/switch.
    d = make("Rollrate")
    d.combo_name.setEditText("Pitchrate")
    assert d._mode() == "rename"
    assert d._base_name == "Rollrate"  # target never switches to Pitchrate

    d._on_confirm()

    assert d.result_action == "cancel"  # blocked


def test_selecting_existing_in_edit_mode_does_not_switch_target(make, no_modals):
    # Picking an existing name from the dropdown only fills the text; it does
    # not silently switch which preset is being edited.
    d = make("Rollrate")
    d.combo_name.setEditText("Pitchrate")  # what a dropdown pick does to the text
    assert d._base_name == "Rollrate"
    assert d.label_old_input.text() == "a/in.0"  # still showing Rollrate
    d._on_confirm()
    assert d.result_action == "cancel"  # collision, not a silent Pitchrate update


def test_add_then_typing_existing_name_stays_create(make, no_modals):
    # In Add mode, typing a name that transiently (or fully) matches an existing
    # preset must NOT turn into an edit/rename.
    d = make(None, create=True)
    d.combo_name.setEditText("Rollrate")  # collides while typing
    assert d._mode() == "create"
    d.combo_name.setEditText("Rollrate2")  # extended to a unique name
    assert d._mode() == "create"
    assert d.btn_confirm.text() == "Create new preset"

    d._on_confirm()

    assert d.result_action == "save"
    assert d.name == "Rollrate2"
    assert d.remove_name is None  # existing "Rollrate" left untouched
    assert d.preset == {"input": SEL_IN, "output": SEL_OUT}


def test_add_then_typing_existing_name_then_confirm_is_rejected(make, no_modals):
    # Leaving an Add-mode name equal to an existing preset is rejected as a
    # collision (rather than silently overwriting).
    d = make(None, create=True)
    d.combo_name.setEditText("Pitchrate")
    assert d._mode() == "create"
    d._on_confirm()
    assert d.result_action == "cancel"


def test_empty_name_is_rejected(make, no_modals):
    d = make(None, create=True)
    d.combo_name.setEditText("")
    d._on_confirm()
    assert d.result_action == "cancel"


def test_delete_targets_selected_preset(make, no_modals):
    d = make("Rollrate")
    d._on_delete()
    assert d.result_action == "delete"
    assert d.remove_name == "Rollrate"
