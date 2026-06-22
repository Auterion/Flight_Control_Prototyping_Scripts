"""Dialog for adding/editing/renaming/deleting input-output presets."""

from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)
from searchable_combo_box import SearchableComboBox


class PresetEditDialog(QDialog):
    """Add or edit a single preset.

    The mode is fixed by the caller via ``create``:
      - create=True: always makes a brand new preset (the name starts blank).
      - create=False: edits ``original_name`` - keeping its name updates its
        signals, typing a brand new name renames it, and it can be deleted.

    The name field autocompletes existing names for convenience, but a name
    that collides with a *different* existing preset is rejected on confirm (it
    never silently overwrites or switches target).

    The "new" input/output default to the current main-window selection.

    After ``exec_()``, read the outcome:
      - result_action == "save":   upsert ``preset`` under ``name``; if
        ``remove_name`` is set and differs from ``name``, delete it first.
      - result_action == "delete": delete preset ``remove_name``.
      - result_action == "cancel": no change.
    """

    def __init__(
        self,
        parent,
        create,
        original_name,
        existing_presets,
        topic_list,
        default_input,
        default_output,
    ):
        super().__init__(parent)
        self.setWindowTitle("Add preset" if create else "Edit preset")

        self.existing = existing_presets

        self.result_action = "cancel"
        self.name = None
        self.preset = None
        self.remove_name = None

        # Fixed at construction. _creating True => create mode (no base preset).
        # Otherwise we edit _base_name. The base never changes while the dialog
        # is open, so autocomplete can't silently switch which preset is edited.
        self._creating = bool(create)
        self._base_name = None if self._creating else original_name

        layout = QVBoxLayout()

        # --- Name (searchable combo of existing presets) ---
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self.combo_name = SearchableComboBox()
        self.combo_name.addItems(list(existing_presets.keys()))
        self.combo_name.setEditText(self._base_name or "")
        self.combo_name.editTextChanged.connect(self._update_mode)
        name_row.addWidget(self.combo_name)
        layout.addLayout(name_row)

        # --- old -> new grid ---
        grid = QGridLayout()
        grid.addWidget(QLabel("<b>Signal</b>"), 0, 0)
        self.label_old_header = QLabel("<b>Old</b>")
        grid.addWidget(self.label_old_header, 0, 1)
        grid.addWidget(QLabel(""), 0, 2)
        grid.addWidget(QLabel("<b>New</b>"), 0, 3)

        grid.addWidget(QLabel("Input:"), 1, 0)
        self.label_old_input = QLabel("")
        grid.addWidget(self.label_old_input, 1, 1)
        self.arrow_input = QLabel("→")
        grid.addWidget(self.arrow_input, 1, 2)
        self.combo_new_input = SearchableComboBox()
        self.combo_new_input.addItems(topic_list)
        self._select(self.combo_new_input, default_input)
        grid.addWidget(self.combo_new_input, 1, 3)

        grid.addWidget(QLabel("Output:"), 2, 0)
        self.label_old_output = QLabel("")
        grid.addWidget(self.label_old_output, 2, 1)
        self.arrow_output = QLabel("→")
        grid.addWidget(self.arrow_output, 2, 2)
        self.combo_new_output = SearchableComboBox()
        self.combo_new_output.addItems(topic_list)
        self._select(self.combo_new_output, default_output)
        grid.addWidget(self.combo_new_output, 2, 3)

        layout.addLayout(grid)

        # --- buttons ---
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.btn_confirm = self.buttons.button(QDialogButtonBox.Ok)
        self.buttons.accepted.connect(self._on_confirm)
        self.buttons.rejected.connect(self.reject)

        self.btn_delete = QPushButton("Delete preset")
        self.btn_delete.setStyleSheet(
            "color: white; background-color: #c0392b; font-weight: bold;"
        )
        self.btn_delete.clicked.connect(self._on_delete)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_delete)
        btn_row.addStretch()
        btn_row.addWidget(self.buttons)
        layout.addLayout(btn_row)

        self.setLayout(layout)

        if self._creating:
            self.combo_name.setFocus()
        self._update_mode()

    def _select(self, combo, text):
        if not text:
            return
        index = combo.findText(text)
        if index >= 0:
            combo.setCurrentIndex(index)
        else:
            combo.setEditText(text)

    def _target_name(self):
        return self.combo_name.currentText().strip()

    def _mode(self):
        """Return 'create', 'update' or 'rename' for the current name.

        Derived from the explicit base preset, never from typed text matching.
        """
        if self._creating or self._base_name is None:
            return "create"
        if self._target_name() == self._base_name:
            return "update"
        return "rename"

    def _source_name(self):
        """Preset whose stored values feed the 'old' column and legacy keys."""
        return self._base_name if self._base_name in self.existing else None

    def _delete_target(self):
        return self._base_name if self._base_name in self.existing else None

    def _update_mode(self):
        mode = self._mode()
        source = self._source_name()
        src = self.existing.get(source, {}) if source else {}

        self.label_old_input.setText(src.get("input", ""))
        self.label_old_output.setText(src.get("output", ""))

        show_old = mode in ("update", "rename")
        for w in (
            self.label_old_header,
            self.label_old_input,
            self.label_old_output,
            self.arrow_input,
            self.arrow_output,
        ):
            w.setVisible(show_old)

        # Editing always reads "Update preset", whether or not the name changed.
        self.btn_confirm.setText(
            "Create new preset" if mode == "create" else "Update preset"
        )

        self.btn_delete.setVisible(self._delete_target() is not None)

    def _on_confirm(self):
        name = self._target_name()
        if not name:
            QMessageBox.warning(
                self, "Invalid name", "The preset name cannot be empty."
            )
            return

        new_input = self.combo_new_input.currentText().strip()
        new_output = self.combo_new_output.currentText().strip()
        if not new_input or not new_output:
            QMessageBox.warning(
                self, "Invalid signals", "Input and output must both be set."
            )
            return

        mode = self._mode()
        if mode in ("create", "rename") and name in self.existing:
            QMessageBox.warning(
                self,
                "Name already exists",
                f"A preset named '{name}' already exists.",
            )
            return

        # Preserve legacy fallbacks from the source preset.
        source = self._source_name()
        src = self.existing.get(source, {}) if source else {}
        preset = {"input": new_input, "output": new_output}
        for key in ("input_legacy", "output_legacy"):
            if key in src:
                preset[key] = src[key]

        self.name = name
        self.preset = preset
        self.remove_name = source if (mode == "rename" and source != name) else None
        self.result_action = "save"
        self.accept()

    def _on_delete(self):
        target = self._delete_target()
        if target is None:
            return
        reply = QMessageBox.question(
            self,
            "Delete preset",
            f"Delete preset '{target}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.remove_name = target
            self.result_action = "delete"
            self.accept()
