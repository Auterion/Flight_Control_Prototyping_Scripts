#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: filter_designer.py
Author: Mathieu Bresciani
Description:
    Standalone GUI to design a chain of digital filters linked in series and
    visualize the combined frequency response (magnitude, phase, group delay).

    Add filters with the "Add filter" button (a popup previews the filter as
    you tune it), then edit/remove each one from the table. Tick a row's "Show"
    box to overlay that individual filter on top of the combined response.

    The heavy lifting lives in filter_library.py (pure numpy/scipy) so the same
    filters can be reused in the control loop of other tools.

Usage:
    python filter_designer.py
"""

import sys

from filter_chain_widget import FilterChainWidget
from PyQt5.QtWidgets import QApplication, QMainWindow


class FilterDesigner(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Digital filter designer")
        self.widget = FilterChainWidget(fs=1000.0)
        self.setCentralWidget(self.widget)
        self.resize(1000, 800)


def main():
    app = QApplication(sys.argv)
    win = FilterDesigner()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
