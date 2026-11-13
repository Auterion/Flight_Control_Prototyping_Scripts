"""Widgets showing a ModelCheck: the verdict, its details and its warning."""

import html

import plot_style as ps
from model_check import (
    hertz,
    kCaution,
    kDoNotTune,
    kFail,
    kGood,
    kPass,
    kUnchecked,
)
from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
)

# A shape per level and state: they do not rely on the color alone
kLevelLabels = {kGood: "✓ Good", kCaution: "⚠ Caution", kDoNotTune: "✕ Do not tune"}
kLevelColors = {
    kGood: ps.GOOD_TEXT,
    kCaution: ps.WARNING_TEXT,
    kDoNotTune: ps.CRITICAL,
}
kStateIcons = {kPass: "✓", kCaution: "⚠", kFail: "✕", kUnchecked: "–"}
kStateColors = {
    kPass: ps.GOOD_TEXT,
    kCaution: ps.WARNING_TEXT,
    kFail: ps.CRITICAL,
    kUnchecked: ps.MUTED,
}
kEvidenceLinks = {
    "bode": "Show this band on the Bode plot",
    "residuals": "Show the residuals",
}
kVerdicts = {
    kGood: "the data confirm the model for the current gains.",
    kCaution: "the model is not verified for the current gains.",
    kDoNotTune: "the data contradict the model for the current gains.",
}
kBannerColors = {
    kCaution: ("#fdf3d7", ps.WARNING),
    kDoNotTune: ("#fbe9e9", ps.CRITICAL),
}

kNameColumnWidth = 170
kPanelSize = (760, 680)

kNoCheckText = (
    "No model checked yet: the check needs data, an identified model and gains."
)
kModelCheckHint = (
    "<b>Can the model be trusted to tune these gains?</b><br>"
    "A controller tuned on a wrong model can be unstable on the vehicle. The "
    "model is compared with the frequency response of the plant measured "
    "directly from the data (the points of the Bode plot), in the band where "
    "the stability margins are decided: around the crossover of the loop. "
    "Changing the gains moves this band, shaded on the Bode plot, so the "
    "verdict follows the gains.<br>"
    "<b>✓ Good:</b> every check passes.<br>"
    "<b>⚠ Caution:</b> a check is borderline or could not be made (–): the "
    "model is not verified.<br>"
    "<b>✕ Do not tune:</b> the data contradict the model.<br><br>"
    "Click the verdict for the details of each check."
)


def verdictSentence(model_check):
    """The verdict and the check that decides it, in one line."""
    sentence = kLevelLabels[model_check.level]
    deciding = model_check.deciding
    return sentence if deciding is None else f"{sentence}: {deciding.reason}"


def bandSentence(model_check):
    """Where the checks look, and why there."""
    critical = model_check.critical
    if critical is None:
        return (
            "With these gains, the loop gain stays far from −1 at all "
            "frequencies: the model is compared with the data over all the "
            "measured frequencies."
        )
    low, high = (hertz(f) for f in model_check.band)
    if critical.is_crossover:
        center = f"the {hertz(critical.f)} Hz crossover of the loop"
    else:
        center = (
            f"{hertz(critical.f)} Hz, where the loop gain is the closest to −1 "
            "(a loop safe at its crossover)"
        )
    return (
        f"The checks look at the band {low}–{high} Hz, from 1/3 to 3 times "
        f"{center}: the stability margins depend on the plant there. Changing "
        "the gains moves this band."
    )


def checksHtml(model_check, with_links=True):
    """The verdict, where the checks look and one table row per check. The
    links name the plot that shows a check that did not pass."""
    level = model_check.level
    rows = []
    for check in model_check.checks:
        details = html.escape(check.value)
        if check.criterion:
            details += (
                f'<br><span style="color:{ps.INK_SECONDARY}">'
                f"{html.escape(check.criterion)}</span>"
            )
        if check.state != kPass:
            if check.hint:
                details += f"<br><b>What to do:</b> {html.escape(check.hint)}"
            if with_links and check.evidence:
                details += (
                    f' <a href="{check.evidence}">{kEvidenceLinks[check.evidence]}</a>'
                )
        rows.append(
            f'<tr><td style="color:{kStateColors[check.state]}">'
            f"<b>{kStateIcons[check.state]}</b></td>"
            f'<td width="{kNameColumnWidth}"><b>{html.escape(check.name)}</b></td>'
            f"<td>{details}</td></tr>"
        )
    return (
        f'<p><b style="color:{kLevelColors[level]}">{kLevelLabels[level]}:</b> '
        f"{kVerdicts[level]}</p>"
        f"<p>{html.escape(bandSentence(model_check))}</p>"
        f'<table cellspacing="0" cellpadding="4">{"".join(rows)}</table>'
    )


class ModelCheckBadge(QPushButton):
    """The verdict, clicked to open its details."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFlat(True)
        self.setCursor(Qt.PointingHandCursor)
        self.showCheck(None)

    def showCheck(self, model_check):
        if model_check is None:
            self.setText("—")
            self.setToolTip(kNoCheckText)
            color = ps.INK
        else:
            self.setText(f"{kLevelLabels[model_check.level]} ▸")
            self.setToolTip(checksHtml(model_check, with_links=False))
            color = kLevelColors[model_check.level]
        self.setStyleSheet(
            "QPushButton { border: none; padding: 0; text-align: left;"
            f" font-weight: bold; color: {color}; }}"
            " QPushButton:hover { text-decoration: underline; }"
        )


class ModelCheckPanel(QDialog):
    """Details of each check. Not modal: it follows the gains while tuning."""

    evidenceRequested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model check")
        self.text = QTextBrowser()
        self.text.setOpenLinks(False)
        self.text.anchorClicked.connect(
            lambda url: self.evidenceRequested.emit(url.toString())
        )
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.text)
        self.setLayout(layout)
        self.resize(*kPanelSize)
        # Placed by its owner when first shown, then left where the user puts it
        self.was_placed = False
        self.showCheck(None)

    def showCheck(self, model_check):
        # A gain change must not scroll the checks being read back to the top
        scroll_bar = self.text.verticalScrollBar()
        position = scroll_bar.value()
        self.text.setHtml(
            kNoCheckText if model_check is None else checksHtml(model_check)
        )
        scroll_bar.setValue(position)


class ElidedLabel(QLabel):
    """One line of text, cut with an ellipsis when the label is too narrow."""

    def paintEvent(self, event):
        text = self.fontMetrics().elidedText(self.text(), Qt.ElideRight, self.width())
        QPainter(self).drawText(self.rect(), Qt.AlignLeft | Qt.AlignVCenter, text)


class ModelCheckBanner(QFrame):
    """One-line warning of a model that is not good, shown where the gains are
    read. Made for the corner of a QTabWidget, which sizes it by its hint."""

    detailsRequested = pyqtSignal()
    overrideRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("modelCheckBanner")
        self.label = ElidedLabel()
        # The text is cut rather than widening the window
        self.label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.btn_details = QPushButton("Details")
        self.btn_details.clicked.connect(self.detailsRequested)
        self.btn_override = QPushButton("Use anyway")
        self.btn_override.setToolTip(
            "Show the gains as usable although the data contradict the model"
        )
        self.btn_override.clicked.connect(self.overrideRequested)
        layout = QHBoxLayout()
        layout.setContentsMargins(8, 1, 4, 1)
        layout.addWidget(self.label, stretch=1)
        for button in (self.btn_details, self.btn_override):
            button.setAutoDefault(False)
            layout.addWidget(button)
        self.setLayout(layout)
        self.hide()

    def showCheck(self, model_check, is_overridden):
        if model_check is None or model_check.level == kGood:
            self.hide()
            return
        level = model_check.level
        background, border = kBannerColors[level]
        self.setStyleSheet(
            f"#modelCheckBanner {{ background: {background};"
            f" border: 1px solid {border}; border-radius: 4px; }}"
        )
        sentence = verdictSentence(model_check)
        self.label.setText(sentence)
        self.label.setToolTip(sentence)
        self.btn_override.setVisible(level == kDoNotTune and not is_overridden)
        self.show()
        self.updateGeometry()

    def sizeHint(self):
        hint = super().sizeHint()
        width = hint.width() + self.label.sizeHint().width()
        tabs = self.parentWidget()
        if tabs is not None and hasattr(tabs, "tabBar"):
            # A wider corner would squeeze the tabs
            width = min(width, tabs.width() - tabs.tabBar().sizeHint().width() - 8)
        return QSize(max(width, hint.width()), hint.height())
