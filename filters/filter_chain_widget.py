#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: filter_chain_widget.py
Author: Mathieu Bresciani
Description:
    Embeddable QWidget presenting a chain of filters linked in series:
      - a global sampling-frequency field,
      - an "Add filter" button,
      - a table with, per row: a "show" checkbox (overlay that filter's
        individual response), a summary, and Edit / Remove buttons,
      - the combined response (always drawn bold) plus every checked filter.

    It is a plain QWidget (not a window) so it can be embedded elsewhere,
    e.g. as a panel/tab of the autotune tool. It emits ``changed`` whenever
    the chain or fs is modified, and exposes ``chain`` (a FilterChain).
"""

from filter_chain_widget_helpers import make_button_cell, make_checkbox_cell
from filter_edit_dialog import FilterEditDialog
from filter_library import FilterChain
from filter_response_canvas import FilterResponseCanvas, Trace
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

COL_SHOW, COL_SUMMARY, COL_EDIT, COL_REMOVE = range(4)

EDIT_LABEL = "⚙"
REMOVE_LABEL = "✕"

# Combined trace is always black; individual filters cycle through this palette
# by their position in the chain (red is reserved for the cursor).
COMBINED_COLOR = "black"
FILTER_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
]


class FilterChainWidget(QWidget):
    """Table of series filters plus their combined frequency response."""

    changed = pyqtSignal()

    def __init__(self, parent=None, fs=1000.0, chain: FilterChain = None):
        super().__init__(parent)
        self.chain = chain if chain is not None else FilterChain()
        self._show_flags = [True] * len(self.chain)

        # --- top row: fs + add ---
        self.spin_fs = QDoubleSpinBox()
        self.spin_fs.setRange(1.0, 1e7)
        self.spin_fs.setDecimals(1)
        self.spin_fs.setValue(fs)
        self.spin_fs.setSuffix(" Hz")
        self.spin_fs.valueChanged.connect(self._on_fs_changed)

        self.btn_add = QPushButton("＋ Add filter")
        self.btn_add.clicked.connect(self._on_add)

        fs_row = QHBoxLayout()
        fs_row.addWidget(QLabel("Sampling freq:"))
        fs_row.addWidget(self.spin_fs)
        fs_row.addStretch()

        # --- table ---
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Show", "Filter", "", ""])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)
        # Default minimum (~36px) would keep the compact icon columns too wide.
        header.setMinimumSectionSize(10)
        # Summary column follows its text; the others are fixed. ResizeToContents
        # measures cell *items* (the summary text) but not embedded widgets, so
        # the widget columns get fixed widths sized to their content instead.
        header.setSectionResizeMode(COL_SUMMARY, QHeaderView.ResizeToContents)
        for col in (COL_SHOW, COL_EDIT, COL_REMOVE):
            header.setSectionResizeMode(col, QHeaderView.Fixed)
        self.table.setColumnWidth(
            COL_SHOW, self.table.fontMetrics().horizontalAdvance("Show") + 16
        )
        self.table.setColumnWidth(
            COL_EDIT, make_button_cell(EDIT_LABEL, compact=True).width() + 2
        )
        self.table.setColumnWidth(
            COL_REMOVE, make_button_cell(REMOVE_LABEL, compact=True).width() + 2
        )

        # --- left panel: fs on top, table, then add button ---
        left = QVBoxLayout()
        left.addLayout(fs_row)
        left.addWidget(self.table)
        left.addWidget(self.btn_add)
        left.addStretch(1)

        # --- plot ---
        self.canvas = FilterResponseCanvas(figsize=(6, 6))

        main = QHBoxLayout(self)
        main.addLayout(left)
        main.addWidget(self.canvas, 1)

        self._rebuild_table()
        self._replot()

    # ------------------------------------------------------------------
    @property
    def fs(self):
        return self.spin_fs.value()

    # --- table construction -------------------------------------------
    def _rebuild_table(self):
        self.table.setRowCount(0)
        for row, flt in enumerate(self.chain):
            self.table.insertRow(row)

            show = make_checkbox_cell(self._show_flags[row])
            show.toggled.connect(
                lambda checked, r=row: self._on_show_toggled(r, checked)
            )
            self.table.setCellWidget(row, COL_SHOW, self._center(show))

            # Type name on the first line, parameters on the second.
            self.table.setItem(
                row, COL_SUMMARY, QTableWidgetItem(f"{flt.name}\n{flt.params_text()}")
            )

            edit = make_button_cell(EDIT_LABEL, compact=True)
            edit.setToolTip("Edit this filter")
            edit.clicked.connect(lambda _, r=row: self._on_edit(r))
            self.table.setCellWidget(row, COL_EDIT, edit)

            remove = make_button_cell(REMOVE_LABEL, danger=True, compact=True)
            remove.setToolTip("Remove this filter")
            remove.clicked.connect(lambda _, r=row: self._on_remove(r))
            self.table.setCellWidget(row, COL_REMOVE, remove)

        self._fit_table_size()

    @staticmethod
    def _center(widget):
        wrap = QWidget()
        lay = QHBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setAlignment(Qt.AlignCenter)
        lay.addWidget(widget)
        return wrap

    def _fit_table_size(self):
        """Size the table to exactly fit its columns and rows (no empty frame)."""
        self.table.resizeColumnToContents(COL_SUMMARY)
        self.table.resizeRowsToContents()
        frame = 2 * self.table.frameWidth()

        width = sum(self.table.columnWidth(c) for c in range(self.table.columnCount()))
        self.table.setFixedWidth(width + frame)

        height = self.table.horizontalHeader().height()
        height += sum(self.table.rowHeight(r) for r in range(self.table.rowCount()))
        self.table.setFixedHeight(height + frame)

    # --- callbacks -----------------------------------------------------
    def _on_fs_changed(self, *_):
        self._replot()
        self.changed.emit()

    def _on_add(self):
        dlg = FilterEditDialog(self, fs=self.fs)
        if dlg.exec_() == FilterEditDialog.Accepted and dlg.result_filter:
            self.chain.add(dlg.result_filter)
            self._show_flags.append(True)
            self._rebuild_table()
            self._replot()
            self.changed.emit()

    def _on_edit(self, row):
        dlg = FilterEditDialog(self, fs=self.fs, flt=self.chain[row].copy())
        if dlg.exec_() == FilterEditDialog.Accepted and dlg.result_filter:
            self.chain.replace(row, dlg.result_filter)
            self._rebuild_table()
            self._replot()
            self.changed.emit()

    def _on_remove(self, row):
        self.chain.remove(row)
        del self._show_flags[row]
        self._rebuild_table()
        self._replot()
        self.changed.emit()

    def _on_show_toggled(self, row, checked):
        self._show_flags[row] = checked
        self._replot()

    # --- plotting ------------------------------------------------------
    def _replot(self):
        traces = []
        for row, flt in enumerate(self.chain):
            if self._show_flags[row]:
                b, a = flt.coefficients(self.fs)
                # Colour is tied to the filter's position in the chain so it
                # stays stable regardless of which filters are shown/hidden.
                color = FILTER_COLORS[row % len(FILTER_COLORS)]
                traces.append(Trace(b, a, label=flt.summary(), bold=False, color=color))
        if len(self.chain) > 0:
            b, a = self.chain.coefficients(self.fs)
            traces.append(
                Trace(b, a, label="Combined", bold=True, color=COMBINED_COLOR)
            )
        self.canvas.plot(traces, self.fs)
