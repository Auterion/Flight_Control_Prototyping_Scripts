#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: filter_response_canvas.py
Author: Mathieu Bresciani
Description:
    Reusable Qt widget showing the Bode-style response (magnitude, phase and
    group delay) of one or more filters. Used by both the add/edit preview
    dialog and the main chain window, and embeddable in other tools.

    Shift + left-click places a vertical red cursor across all three axes. It
    reads the magnitude / phase / group-delay values off the combined (bold)
    trace at that frequency, annotates each plot, and labels the exact cursor
    frequency under the bottom (group-delay) x-axis. The cursor persists across
    redraws (e.g. when a filter parameter changes).
"""

import numpy as np
from filter_library import frequency_response, group_delay_ms
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.transforms import blended_transform_factory
from PyQt5.QtWidgets import QVBoxLayout, QWidget

CURSOR_COLOR = "red"


class Trace:
    """One curve to draw: coefficients (b, a) plus how to style it."""

    def __init__(self, b, a, label="", bold=False, color=None):
        self.b = b
        self.a = a
        self.label = label
        self.bold = bold
        self.color = color


class FilterResponseCanvas(QWidget):
    """Three stacked, x-shared log-frequency axes: magnitude / phase / delay."""

    def __init__(self, parent=None, figsize=(6, 6)):
        super().__init__(parent)
        self.figure = Figure(figsize=figsize, constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax_mag = self.figure.add_subplot(3, 1, 1)
        self.ax_phase = self.figure.add_subplot(3, 1, 2, sharex=self.ax_mag)
        self.ax_gd = self.figure.add_subplot(3, 1, 3, sharex=self.ax_mag)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        # Cursor state.
        self._fs = 0.0
        self._cursor_freq = None  # None => no cursor
        self._cursor_artists = []
        self._primary = None  # lookup arrays of the trace the cursor reads
        self._dragging = False

        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def plot(self, traces, fs):
        """Draw the given traces. ``traces`` is an iterable of ``Trace``."""
        self._fs = fs
        # Axes are about to be cleared: their old cursor artists die with them.
        self._cursor_artists = []
        self._primary = None

        for ax in (self.ax_mag, self.ax_phase, self.ax_gd):
            # Reset to linear first: clearing a log axis warns while it
            # momentarily resets the limits to the invalid (0, 1).
            ax.set_xscale("linear")
            ax.clear()

        any_labelled = False
        for tr in traces:
            lw = 2.4 if tr.bold else 1.2
            alpha = 1.0 if tr.bold else 0.7
            w, mag, phase = frequency_response(tr.b, tr.a, fs)
            wg, gd = group_delay_ms(tr.b, tr.a, fs)
            # Drop the DC (0 Hz) bin: it cannot be shown on a log axis.
            s = slice(1, None)
            (line,) = self.ax_mag.semilogx(
                w[s],
                mag[s],
                linewidth=lw,
                alpha=alpha,
                color=tr.color,
                label=tr.label or None,
            )
            color = line.get_color()
            self.ax_phase.semilogx(
                w[s], phase[s], linewidth=lw, alpha=alpha, color=color
            )
            self.ax_gd.semilogx(wg[s], gd[s], linewidth=lw, alpha=alpha, color=color)
            any_labelled = any_labelled or bool(tr.label)

            # The cursor reads off the bold (combined) trace, or the first one.
            if self._primary is None or tr.bold:
                self._primary = {"w": w, "mag": mag, "phase": phase, "wg": wg, "gd": gd}

        self.ax_mag.set_ylabel("Amplitude (dB)")
        self.ax_phase.set_ylabel("Phase (deg)")
        self.ax_gd.set_ylabel("Group delay (ms)")
        self.ax_gd.set_xlabel("Frequency (Hz)")
        for ax in (self.ax_mag, self.ax_phase, self.ax_gd):
            ax.grid(True, which="both", alpha=0.3)
        if fs > 2.0:
            self.ax_mag.set_xlim(left=1.0, right=fs / 2.0)
        if any_labelled:
            self.ax_mag.legend(fontsize=8, loc="lower left")

        self._draw_cursor()

    def clear(self):
        self._cursor_artists = []
        self._primary = None
        for ax in (self.ax_mag, self.ax_phase, self.ax_gd):
            ax.set_xscale("linear")
            ax.clear()
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Cursor
    # ------------------------------------------------------------------
    def _on_press(self, event):
        # Any left-click (with or without shift) places the cursor and starts
        # a drag.
        if event.button != 1 or event.inaxes is None or event.xdata is None:
            return
        self._dragging = True
        self._set_cursor(event.xdata)

    def _on_motion(self, event):
        # Keep tracking while the button is held.
        if not self._dragging or event.inaxes is None or event.xdata is None:
            return
        self._set_cursor(event.xdata)

    def _on_release(self, event):
        if event.button == 1:
            self._dragging = False

    def _set_cursor(self, xdata):
        freq = float(xdata)
        if freq <= 0.0:
            return
        self._cursor_freq = freq
        self._draw_cursor()

    def _remove_cursor_artists(self):
        for art in self._cursor_artists:
            try:
                art.remove()
            except (ValueError, NotImplementedError):
                pass
        self._cursor_artists = []

    def _draw_cursor(self):
        self._remove_cursor_artists()
        freq = self._cursor_freq
        if freq is None or self._primary is None:
            self.canvas.draw_idle()
            return

        p = self._primary
        mag = float(np.interp(freq, p["w"], p["mag"]))
        phase = float(np.interp(freq, p["w"], p["phase"]))
        gd = float(np.interp(freq, p["wg"], p["gd"]))

        readouts = [
            (self.ax_mag, mag, f"{mag:.2f} dB"),
            (self.ax_phase, phase, f"{phase:.1f}°"),
            (self.ax_gd, gd, f"{gd:.2f} ms"),
        ]
        for ax, yval, text in readouts:
            self._cursor_artists.append(
                ax.axvline(freq, color=CURSOR_COLOR, linewidth=1.0, alpha=0.9)
            )
            self._cursor_artists.append(
                ax.plot([freq], [yval], "o", color=CURSOR_COLOR, markersize=4)[0]
            )
            self._cursor_artists.append(
                ax.annotate(
                    text,
                    xy=(freq, yval),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    color=CURSOR_COLOR,
                    ha="left",
                    va="bottom",
                )
            )

        # Exact frequency, labelled under the bottom (group-delay) x-axis.
        trans = blended_transform_factory(self.ax_gd.transData, self.ax_gd.transAxes)
        self._cursor_artists.append(
            self.ax_gd.annotate(
                f"{freq:.4g} Hz",
                xy=(freq, 0.0),
                xycoords=trans,
                xytext=(0, -18),
                textcoords="offset points",
                fontsize=8,
                color=CURSOR_COLOR,
                ha="center",
                va="top",
            )
        )

        self.canvas.draw_idle()
