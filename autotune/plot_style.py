"""pyqtgraph styling and plot helpers shared by the autotune windows.

Colors are the light steps of the dataviz reference palette, validated
against SURFACE.
"""

import pyqtgraph as pg
from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtWidgets import QGraphicsTextItem
from pyqtgraph.graphicsItems.GraphicsWidgetAnchor import GraphicsWidgetAnchor

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Categorical slots, assigned in this order
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
VIOLET = "#6250d6"
RED = "#e34948"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]

GOOD = "#0ca30c"
# The status "good" step is too light for text on SURFACE
GOOD_TEXT = "#006300"
CRITICAL = "#d03b3b"

# The options apply to widgets created afterwards: set them on import, before
# any window builds its plots
pg.setConfigOptions(background=SURFACE, foreground=INK, antialias=True)

_PEN_STYLES = {
    "-": Qt.SolidLine,
    "--": Qt.DashLine,
    ":": Qt.DotLine,
}


def color(value, alpha=1.0):
    c = pg.mkColor(value)
    c.setAlphaF(alpha)
    return c


def pen(value, width=1.0, style="-", alpha=1.0):
    p = pg.mkPen(color(value, alpha), width=width, style=_PEN_STYLES[style])
    p.setCosmetic(True)
    return p


def stylePlot(plot, title=None, xlabel=None, ylabel=None):
    for name in ("left", "bottom"):
        axis = plot.getAxis(name)
        axis.setPen(pg.mkPen(AXIS))
        axis.setTextPen(pg.mkPen(INK_SECONDARY))
    if title is not None:
        plot.setTitle(title, color=INK, size="10pt")
    if xlabel is not None:
        plot.setLabel("bottom", xlabel, color=INK_SECONDARY)
    if ylabel is not None:
        plot.setLabel("left", ylabel, color=INK_SECONDARY)


def addLegend(plot, corner="top-right"):
    offsets = {
        "top-right": (-5, 5),
        "top-left": (5, 5),
        "bottom-right": (-5, -5),
        "bottom-left": (5, -5),
    }
    return plot.addLegend(
        offset=offsets[corner],
        brush=color(SURFACE, 0.8),
        labelTextColor=INK_SECONDARY,
        labelTextSize="8pt",
        verSpacing=-5,
    )


def addReferenceLine(plot, value, angle, color_=AXIS, style="-", width=1.0):
    line = pg.InfiniteLine(value, angle=angle, pen=pen(color_, width, style))
    plot.addItem(line, ignoreBounds=True)
    return line


class ResettableViewBox(pg.ViewBox):
    """Double-click restores the default range, or autorange if there is none.

    The default range is in view coordinates: log10 of the data on a log axis.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_x_range = None
        self.default_y_range = None
        self.has_default_range = False

    def setDefaultRange(self, x_range=None, y_range=None):
        """Apply the new default range when the X default changes.

        This keeps the user's zoom across data updates, but resets the view
        when what is plotted changes scale (a new sample time, for example).
        A Y default that follows the data (a peak, say) is only applied by
        the next reset.
        """
        changed = not self.has_default_range or x_range != self.default_x_range
        self.default_x_range = x_range
        self.default_y_range = y_range
        self.has_default_range = True
        if changed:
            self.resetRange()

    def resetRange(self):
        if self.default_x_range is None:
            self.enableAutoRange(x=True)
        else:
            self.setXRange(*self.default_x_range, padding=0)
        if self.default_y_range is None:
            self.enableAutoRange(y=True)
        else:
            self.setYRange(*self.default_y_range, padding=0)

    def mouseClickEvent(self, ev):
        if ev.double() and ev.button() == Qt.LeftButton:
            ev.accept()
            self.resetRange()
        else:
            super().mouseClickEvent(ev)


def addPlot(layout, row, col, rowspan=1, colspan=1, **style):
    plot = layout.addPlot(
        row=row,
        col=col,
        rowspan=rowspan,
        colspan=colspan,
        viewBox=ResettableViewBox(),
    )
    plot.hideButtons()
    stylePlot(plot, **style)
    return plot


def lockAspect(plot):
    plot.setAspectLocked(True)
    # The locked range follows the view width and the view width follows the
    # tick labels: an axis that shrinks back can flicker between two tick
    # spacings forever
    plot.getAxis("left").setStyle(autoReduceTextSpace=False)


def linkXRanges(view_a, view_b):
    """Keep the X ranges of two views equal, even across tabs."""

    def follow(source, target):
        def onRangeChanged(_, x_range):
            if target.viewRange()[0] != list(x_range):
                target.setXRange(*x_range, padding=0)

        source.sigXRangeChanged.connect(onRangeChanged)

    follow(view_a, view_b)
    follow(view_b, view_a)


def setLegendEntry(legend, item, name):
    """Show item in legend under name, or remove it when name is None."""
    entry = next((label for sample, label in legend.items if sample.item is item), None)
    if name is None:
        if entry is not None:
            legend.removeItem(item)
    elif entry is None:
        legend.addItem(item, name)
    else:
        entry.setText(name)


def xOnlyMouse(plot):
    """Wheel and drag act on X; Y fits the data visible in the X range."""
    plot.setMouseEnabled(x=True, y=False)
    plot.enableAutoRange(y=True)
    plot.setAutoVisible(y=True)


class CornerText(GraphicsWidgetAnchor, pg.GraphicsWidget):
    """Lines of colored text pinned to a corner of a plot, on a translucent box.

    It stays in place when the view pans or zooms.
    """

    _CORNERS = {
        "top-left": ((0, 0), (5, 5)),
        "top-right": ((1, 0), (-5, 5)),
        "bottom-left": ((0, 1), (5, -5)),
        "bottom-right": ((1, 1), (-5, -5)),
    }

    def __init__(self, plot, corner="top-left", size="8pt"):
        pg.GraphicsWidget.__init__(self)
        GraphicsWidgetAnchor.__init__(self)
        self.size = size
        self.text_item = QGraphicsTextItem(self)
        self.text_item.setPos(2, 0)
        self.setParentItem(plot.getViewBox())
        self.setZValue(100)
        anchor, offset = self._CORNERS[corner]
        self.anchor(itemPos=anchor, parentPos=anchor, offset=offset)
        self.setLines([])

    def setLines(self, lines):
        """lines: (text, color) pairs; a None color uses the primary ink."""
        html = "<br>".join(
            f"<span style='color: {color_ or INK}; font-size: {self.size}'>{text}</span>"
            for text, color_ in lines
            if text
        )
        self.text_item.setHtml(html)
        self.setVisible(bool(html))
        rect = self.text_item.boundingRect()
        self.resize(rect.width() + 4, rect.height())

    def text(self):
        return self.text_item.toPlainText()

    def boundingRect(self):
        return QRectF(0, 0, self.width(), self.height())

    def paint(self, painter, *args):
        painter.setPen(Qt.NoPen)
        painter.setBrush(color(SURFACE, 0.8))
        painter.drawRect(self.boundingRect())


class HoverReadout:
    """Crosshair and x/y readout under the mouse, in data units on log axes."""

    def __init__(self, plot, x_format="x = {:.3g}", y_format="y = {:.3g}"):
        self.plot = plot
        self.x_format = x_format
        self.y_format = y_format
        crosshair_pen = pen(MUTED, 1, "--")
        self.v_line = pg.InfiniteLine(angle=90, pen=crosshair_pen)
        self.h_line = pg.InfiniteLine(angle=0, pen=crosshair_pen)
        for line in (self.v_line, self.h_line):
            plot.addItem(line, ignoreBounds=True)
        self.label = CornerText(plot, "bottom-right")
        self.show(False)
        plot.scene().sigMouseMoved.connect(self.onMouseMoved)

    def show(self, visible):
        for item in (self.v_line, self.h_line, self.label):
            item.setVisible(visible)

    def onMouseMoved(self, scene_pos):
        view_box = self.plot.getViewBox()
        if not view_box.sceneBoundingRect().contains(scene_pos):
            self.show(False)
            return
        point = view_box.mapSceneToView(QPointF(scene_pos))
        self.v_line.setPos(point.x())
        self.h_line.setPos(point.y())
        x_log, y_log = view_box.state["logMode"]
        x = 10 ** point.x() if x_log else point.x()
        y = 10 ** point.y() if y_log else point.y()
        self.label.setLines(
            [(f"{self.x_format.format(x)}, {self.y_format.format(y)}", None)]
        )
        self.show(True)
