"""Autotune sessions: the state of the interface saved by group in a YAML file.

A session holds the groups chosen when saving, each a top-level key; a group
that was not saved is absent. The data group references the log by its path
relative to the session file, its absolute path and its SHA-256, the log
itself is never stored.
"""

import hashlib
import os

import yaml
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QLabel,
    QVBoxLayout,
)

kSessionFormat = "px4-autotune-session"
kSessionVersion = 1
kSessionFileFilter = "Autotune session (*.yaml)"
kSessionGroups = {
    "data": "Data",
    "preprocessing": "Pre-processing",
    "model_structure": "Model structure",
    "model": "Model result",
    "tuning": "Tuning",
    "others": "Others",
}


class SessionError(Exception):
    pass


def fileSha256(path):
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def logEntry(log_path, session_path):
    log_path = os.path.abspath(log_path)
    session_dir = os.path.dirname(os.path.abspath(session_path))
    try:
        common = os.path.commonpath([log_path, session_dir])
    except ValueError:
        # On another Windows drive than the session
        common = None
    path = log_path
    # Up from the root, a relative path is only a longer absolute one
    if common is not None and os.path.dirname(common) != common:
        path = os.path.relpath(log_path, session_dir)
    return {"path": path, "absolute": log_path, "sha256": fileSha256(log_path)}


def saveSession(path, groups):
    """Write the states of groups, data["log"] being the path of the log."""
    session = {"format": kSessionFormat, "version": kSessionVersion, **groups}
    if "data" in groups:
        session["data"] = {
            **groups["data"],
            "log": logEntry(groups["data"]["log"], path),
        }
    with open(path, "w") as f:
        yaml.safe_dump(session, f, sort_keys=False)


def loadSession(path):
    try:
        with open(path) as f:
            session = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        raise SessionError(f"Cannot read {path}: {e}")
    if not isinstance(session, dict) or session.get("format") != kSessionFormat:
        raise SessionError(f"{path} is not an autotune session.")
    if session.get("version", 0) > kSessionVersion:
        raise SessionError(
            f"{path} was saved by a newer version of autotune "
            f"(session version {session['version']}, this one reads up to "
            f"{kSessionVersion})."
        )
    return session


def resolveLogPath(session_path, log_entry):
    """Path of the log of a session, None when it is not found.

    Looks relative to the session file first, so that a session moved with
    its log still finds it, then at the absolute path, then for the log's
    name next to the session.
    """
    session_dir = os.path.dirname(os.path.abspath(session_path))
    candidates = [
        os.path.join(session_dir, log_entry.get("path", "")),
        log_entry.get("absolute", ""),
        os.path.join(session_dir, os.path.basename(log_entry.get("path", ""))),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.normpath(candidate)
    return None


def isSameLog(path, log_entry):
    return "sha256" not in log_entry or fileSha256(path) == log_entry["sha256"]


def withDefaults(loaded, defaults):
    """loaded completed with the values of defaults it misses, in nested
    dicts too: a session lacking a key keeps the current value."""
    if not isinstance(loaded, dict) or not isinstance(defaults, dict):
        return loaded
    merged = dict(defaults)
    for key, value in loaded.items():
        merged[key] = withDefaults(value, defaults.get(key))
    return merged


def counted(count, noun):
    return f"{count} {noun}{'' if count == 1 else 's'}"


def formatRange(t_range):
    return f"{t_range[0]:.1f}–{t_range[1]:.1f} s"


def groupSummary(group, state):
    """One line describing the state of group, for the group dialog."""
    if group == "data":
        log = state.get("log")
        if isinstance(log, dict):
            log = log.get("path", "")
        preset = (state.get("preset") or {}).get("name", "no preset")
        estimation = state.get("estimation_range")
        validation = state.get("validation_range")
        parts = [
            os.path.basename(log or ""),
            preset,
            formatRange(estimation) if estimation else "whole log",
        ]
        if validation:
            parts.append(f"validation {formatRange(validation)}")
        return ", ".join(parts)
    if group == "preprocessing":
        offset = "offset removed" if state.get("remove_offset") else "offset kept"
        return (
            f"{offset}, HP {state.get('hp_cutoff')} Hz, LP {state.get('lp_cutoff')} "
            f"Hz, input scaling {state.get('input_scaling')}"
        )
    if group == "model_structure":
        return (
            f"{counted(state.get('poles'), 'pole')}, "
            f"{counted(state.get('zeros'), 'zero')}, "
            f"{counted(state.get('delays'), 'delay')}, {state.get('method')}, "
            f"prediction horizon {state.get('prediction_horizon')} s"
        )
    if group == "model":
        return (
            f"{counted(len(state.get('den', [])) - 1, 'pole')}, "
            f"{counted(len(state.get('num', [])) - 1, 'zero')}, "
            f"{counted(state.get('delays'), 'delay')}, dt {state.get('dt')} s"
        )
    if group == "tuning":
        gains = ", ".join(
            f"{gain} {value:.4g}" for gain, value in state.get("gains", {}).items()
        )
        return (
            f"{state.get('controller_tab')} tab, {state.get('gain_form')} form, "
            f"ideal gains {gains}"
        )
    if group == "others":
        return "step info limits, step simulation, model output offset"
    return ""


class SessionGroupsDialog(QDialog):
    """Choice of the groups to save or load.

    states maps each group to its state, None for a group that cannot be
    chosen, which is then shown with missing_text.
    """

    def __init__(self, parent, title, states, missing_text, notes=""):
        super().__init__(parent)
        self.setWindowTitle(title)
        layout = QVBoxLayout(self)
        self.all_box = QCheckBox("All")
        self.all_box.setTristate(True)
        self.all_box.clicked.connect(self.checkAllGroups)
        layout.addWidget(self.all_box)
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)
        self.group_boxes = {}
        for group, label in kSessionGroups.items():
            state = states.get(group)
            box = QCheckBox(label)
            box.setChecked(state is not None)
            box.setEnabled(state is not None)
            box.toggled.connect(self.onGroupToggled)
            self.group_boxes[group] = box
            summary = QLabel(
                missing_text if state is None else groupSummary(group, state)
            )
            summary.setWordWrap(True)
            summary.setStyleSheet("color: gray; margin-left: 20px")
            layout.addWidget(box)
            layout.addWidget(summary)
        if notes:
            notes_label = QLabel(notes)
            notes_label.setWordWrap(True)
            layout.addWidget(notes_label)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.onGroupToggled()

    def selectedGroups(self):
        return [group for group, box in self.group_boxes.items() if box.isChecked()]

    def choosableBoxes(self):
        return [box for box in self.group_boxes.values() if box.isEnabled()]

    def checkAllGroups(self):
        # A click on a partially checked box makes it checked: all groups
        checked = self.all_box.checkState() != Qt.Unchecked
        for box in self.choosableBoxes():
            box.setChecked(checked)

    def onGroupToggled(self):
        n_selected = len(self.selectedGroups())
        n_choosable = len(self.choosableBoxes())
        if n_selected == 0:
            state = Qt.Unchecked
        elif n_selected == n_choosable:
            state = Qt.Checked
        else:
            state = Qt.PartiallyChecked
        self.all_box.setCheckState(state)
        self.all_box.setEnabled(n_choosable > 0)
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(n_selected > 0)
