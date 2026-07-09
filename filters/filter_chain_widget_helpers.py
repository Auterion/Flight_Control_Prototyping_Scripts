#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: filter_chain_widget_helpers.py
Author: Mathieu Bresciani
Description:
    Small widget factory helpers for the filter chain table.
"""

from PyQt5.QtWidgets import QCheckBox, QPushButton


def make_checkbox_cell(checked=False):
    box = QCheckBox()
    box.setChecked(checked)
    box.setToolTip("Overlay this filter's individual response")
    return box


def make_button_cell(text, danger=False, compact=False):
    btn = QPushButton(text)
    if danger:
        btn.setStyleSheet("color: white; background-color: #c0392b;")
    # Reserve room for the label up front: ResizeToContents otherwise measures
    # the column before the embedded button has laid out and clips the text.
    # Icon buttons (a single glyph) get tight padding so they stay small.
    text_width = btn.fontMetrics().horizontalAdvance(text)
    padding = 10 if compact else 24
    if compact:
        btn.setFixedWidth(text_width + padding)
    else:
        btn.setMinimumWidth(text_width + padding)
    return btn
