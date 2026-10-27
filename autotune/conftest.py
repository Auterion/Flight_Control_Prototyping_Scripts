import pytest
from PyQt5.QtWidgets import QApplication


@pytest.fixture(autouse=True)
def close_windows():
    yield
    # A window left open is painted again while the garbage collector tears
    # down its pyqtgraph scene, which segfaults in GraphicsView.paintEvent
    app = QApplication.instance()
    if app is not None:
        for widget in app.topLevelWidgets():
            widget.close()
