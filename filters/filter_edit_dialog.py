#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: filter_edit_dialog.py
Author: Mathieu Bresciani
Description:
    Modal dialog to add or edit a single filter. A combo box selects the
    filter type; parameter fields are rebuilt from the type's ParamSpec list.
    An embedded FilterResponseCanvas previews the filter live as parameters
    change.
"""

from filter_library import FILTER_TYPE_IDS, FILTER_TYPES, Filter
from filter_response_canvas import FilterResponseCanvas, Trace
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
)


class FilterEditDialog(QDialog):
    """Add (``flt=None``) or edit an existing filter.

    After ``exec_()`` returns ``QDialog.Accepted``, read ``self.result_filter``.
    """

    def __init__(self, parent=None, fs=1000.0, flt: Filter = None):
        super().__init__(parent)
        self.setWindowTitle("Add filter" if flt is None else "Edit filter")
        self.fs = fs
        self.result_filter = None
        self._param_widgets = {}  # key -> QDoubleSpinBox

        # --- filter type combo ---
        self.combo_type = QComboBox()
        for tid in FILTER_TYPE_IDS:
            self.combo_type.addItem(FILTER_TYPES[tid].name, tid)

        # --- dynamic parameter form ---
        self.param_group = QGroupBox("Parameters")
        self.param_form = QFormLayout(self.param_group)

        # --- preview ---
        self.canvas = FilterResponseCanvas(figsize=(5, 6))

        # --- buttons ---
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        left = QVBoxLayout()
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Filter type:"))
        type_row.addWidget(self.combo_type, 1)
        left.addLayout(type_row)
        left.addWidget(self.param_group)
        left.addStretch()
        left.addWidget(buttons)

        right = QVBoxLayout()
        right.addWidget(QLabel("Preview"))
        right.addWidget(self.canvas, 1)

        main = QHBoxLayout(self)
        main.addLayout(left)
        main.addLayout(right, 1)

        # Preselect type / params when editing.
        if flt is not None:
            index = self.combo_type.findData(flt.type_id)
            if index >= 0:
                self.combo_type.setCurrentIndex(index)
        self.combo_type.currentIndexChanged.connect(self._rebuild_params)
        self._rebuild_params(preset_params=flt.params if flt else None)

    # ------------------------------------------------------------------
    def _current_type(self):
        return FILTER_TYPES[self.combo_type.currentData()]

    def _rebuild_params(self, *_, preset_params=None):
        # Drop existing rows.
        while self.param_form.rowCount():
            self.param_form.removeRow(0)
        self._param_widgets.clear()

        for spec in self._current_type().params:
            spin = QDoubleSpinBox()
            spin.setRange(spec.minimum, spec.maximum)
            spin.setDecimals(spec.decimals)
            spin.setSingleStep(max(spec.minimum, 1.0))
            value = (
                preset_params.get(spec.key, spec.default)
                if preset_params
                else spec.default
            )
            spin.setValue(value)
            if spec.unit:
                spin.setSuffix(f" {spec.unit}")
            spin.valueChanged.connect(self._update_preview)
            self._param_widgets[spec.key] = spin
            self.param_form.addRow(spec.label + ":", spin)

        self._update_preview()

    def _collect_params(self):
        return {key: w.value() for key, w in self._param_widgets.items()}

    def _build_filter(self):
        return Filter(self.combo_type.currentData(), self._collect_params())

    def _update_preview(self, *_):
        try:
            flt = self._build_filter()
            b, a = flt.coefficients(self.fs)
            self.canvas.plot([Trace(b, a, bold=True)], self.fs)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            # Invalid parameter combination mid-edit; skip this redraw.
            self.canvas.clear()

    def _on_accept(self):
        self.result_filter = self._build_filter()
        self.accept()
