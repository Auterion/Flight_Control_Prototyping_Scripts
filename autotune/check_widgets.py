"""Widgets showing the model check and the tuning check: their verdicts,
their details and their warning."""

import html

import plot_style as ps
from checks import hertz, kBad, kCaution, kFail, kGood, kPass, kUnchecked
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
    QWidget,
)

kModel = "model"
kTuning = "tuning"

# A shape per level and state: they do not rely on the color alone
kLevelLabels = {
    kModel: {kGood: "✓ Good", kCaution: "⚠ Caution", kBad: "✕ Bad"},
    kTuning: {kGood: "✓ Good", kCaution: "⚠ Low margins", kBad: "✕ Unstable"},
}
kVerdicts = {
    kModel: {
        kGood: "the data confirm the model for the current gains.",
        kCaution: "the model is not verified for the current gains.",
        kBad: "the data contradict the model for the current gains.",
    },
    kTuning: {
        kGood: "the current gains keep the recommended stability margins.",
        kCaution: "the current gains leave too little stability margin.",
        kBad: "the loop is unstable with the current gains.",
    },
}
kLevelColors = {kGood: ps.GOOD_TEXT, kCaution: ps.WARNING_TEXT, kBad: ps.CRITICAL}
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
    "robustness": "Show the margins on the Nyquist plot",
}
# Of a note on the tool, as the upgrade banner: not a state of the checks
kDisclaimerColors = ("#e8f1fc", ps.BLUE)
kBannerColors = {
    kCaution: ("#fdf3d7", ps.WARNING),
    kBad: ("#fbe9e9", ps.CRITICAL),
}

kNameColumnWidth = 170
kPanelSize = (760, 680)

kNoModelCheckText = (
    "No model checked yet: the check needs data, an identified model and gains."
)
kNoTuningCheckText = "No tuning checked yet: the check needs a model and gains."
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
    "<b>✕ Bad:</b> the data contradict the model.<br><br>"
    "Click the verdict for the details of each check."
)
kChecksDisclaimer = (
    "These checks are automated and based on good-practice rules. They do not "
    "guarantee a stable or good behavior on the actual vehicle."
)
kTuningNote = (
    "The margins are those of the loop closed on the model: they hold as far "
    "as the model does, which the model check tells."
)


def verdictSentence(kind, verdict):
    """The verdict and the check that decides it, in one line."""
    sentence = kLevelLabels[kind][verdict.level]
    deciding = verdict.deciding
    return sentence if deciding is None else f"{sentence}: {deciding.reason}"


def bandSentence(model_check):
    """Where the checks of the model look, and why there."""
    critical = model_check.critical
    if critical is None:
        return (
            "With these gains, the loop gain stays far from −1 at all "
            "frequencies: the model is compared with the data over all the "
            "measured frequencies, not only over the band shaded on the Bode "
            "plot."
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


def verdictHtml(kind, verdict, note, with_links):
    """The verdict, a note on its checks and one table row per check. The
    links name the plot that shows a check that did not pass."""
    level = verdict.level
    rows = []
    for check in verdict.checks:
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
        f'<p><b style="color:{kLevelColors[level]}">{kLevelLabels[kind][level]}:'
        f"</b> {kVerdicts[kind][level]}</p>"
        f"<p>{html.escape(note)}</p>"
        f'<table cellspacing="0" cellpadding="4">{"".join(rows)}</table>'
    )


def modelCheckHtml(model_check, with_links=True):
    if model_check is None:
        return f"<p>{kNoModelCheckText}</p>"
    return verdictHtml(kModel, model_check, bandSentence(model_check), with_links)


def tuningCheckHtml(tuning_check, with_links=True):
    if tuning_check is None:
        return f"<p>{kNoTuningCheckText}</p>"
    return verdictHtml(kTuning, tuning_check, kTuningNote, with_links)


def checksHtml(model_check, tuning_check):
    """Both checks, each under a heading named as its kind to scroll to."""
    return (
        f'<h3><a name="{kModel}">Model check</a></h3>{modelCheckHtml(model_check)}'
        f'<h3><a name="{kTuning}">Tuning check</a></h3>'
        f"{tuningCheckHtml(tuning_check)}"
    )


class CheckBadge(QPushButton):
    """The verdict of a check of a kind, clicked to open its details."""

    def __init__(self, kind, prefix="", parent=None):
        super().__init__(parent)
        self.kind = kind
        self.prefix = prefix
        self.setFlat(True)
        self.setCursor(Qt.PointingHandCursor)
        # Not by the style sheet: the size hint measures the texts with it
        font = self.font()
        font.setBold(True)
        self.setFont(font)
        self.showCheck(None)

    def verdictText(self, level):
        return f"{self.prefix}{kLevelLabels[self.kind][level]} ▸"

    def showCheck(self, verdict):
        if self.kind == kModel:
            self.setToolTip(modelCheckHtml(verdict, with_links=False))
        else:
            self.setToolTip(tuningCheckHtml(verdict, with_links=False))
        if verdict is None:
            self.setText(f"{self.prefix}—")
            color = ps.INK
        else:
            self.setText(self.verdictText(verdict.level))
            color = kLevelColors[verdict.level]
        self.setStyleSheet(
            "QPushButton { border: none; padding: 0; text-align: left;"
            f" color: {color}; }}"
            " QPushButton:hover { text-decoration: underline; }"
        )

    def sizeHint(self):
        """As wide as its widest verdict: a change of verdict must not move
        the layout around the badge."""
        hint = super().sizeHint()
        widest = max(
            self.textWidth(self.verdictText(level)) for level in kLevelLabels[self.kind]
        )
        return QSize(hint.width() - self.textWidth(self.text()) + widest, hint.height())

    def textWidth(self, text):
        # As QPushButton measures its text for its size hint
        return self.fontMetrics().size(Qt.TextShowMnemonic, text).width()


class ChecksPanel(QDialog):
    """Details of each check of the model and of the tuning. Not modal: it
    follows the gains while tuning."""

    evidenceRequested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model and tuning checks")
        self.text = QTextBrowser()
        self.text.setOpenLinks(False)
        self.text.anchorClicked.connect(
            lambda url: self.evidenceRequested.emit(url.toString())
        )
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        # Over the text, not in it: it stays in view whatever the scroll
        layout.addWidget(self.createDisclaimer())
        layout.addWidget(self.text)
        self.setLayout(layout)
        self.resize(*kPanelSize)
        # Placed by its owner when first shown, then left where the user puts it
        self.was_placed = False
        self.showChecks(None, None)

    def createDisclaimer(self):
        self.disclaimer = QLabel(f"ⓘ {kChecksDisclaimer}")
        self.disclaimer.setWordWrap(True)
        frame = QFrame()
        frame.setObjectName("checksDisclaimer")
        background, border = kDisclaimerColors
        frame.setStyleSheet(
            f"#checksDisclaimer {{ background: {background};"
            f" border-bottom: 1px solid {border}; }}"
        )
        layout = QHBoxLayout()
        layout.setContentsMargins(8, 6, 8, 6)
        layout.addWidget(self.disclaimer)
        frame.setLayout(layout)
        return frame

    def showChecks(self, model_check, tuning_check):
        # A gain change must not scroll the checks being read back to the top
        scroll_bar = self.text.verticalScrollBar()
        position = scroll_bar.value()
        self.text.setHtml(checksHtml(model_check, tuning_check))
        scroll_bar.setValue(position)

    def showSection(self, kind):
        self.text.scrollToAnchor(kind)


class ElidedLabel(QLabel):
    """One line of text, cut with an ellipsis when the label is too narrow."""

    def paintEvent(self, event):
        text = self.fontMetrics().elidedText(self.text(), Qt.ElideRight, self.width())
        QPainter(self).drawText(self.rect(), Qt.AlignLeft | Qt.AlignVCenter, text)


class ChecksBanner(QFrame):
    """One-line warning of the checks that are not good."""

    # Kind of the first check that is not good
    detailsRequested = pyqtSignal(str)
    overrideRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("checksBanner")
        self.warned_kind = kModel
        self.label = ElidedLabel()
        # The text is cut rather than widening the window
        self.label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.btn_details = QPushButton("Details")
        self.btn_details.clicked.connect(
            lambda: self.detailsRequested.emit(self.warned_kind)
        )
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

    def showChecks(self, model_check, tuning_check, is_overridden):
        warned = [
            (kind, verdict)
            for kind, verdict in ((kModel, model_check), (kTuning, tuning_check))
            if verdict is not None and verdict.level != kGood
        ]
        if not warned:
            self.hide()
            return
        # The worst first: the end of the line is cut in a narrow window
        warned.sort(key=lambda kind_verdict: kind_verdict[1].level != kBad)
        self.warned_kind = warned[0][0]
        is_bad = any(verdict.level == kBad for _, verdict in warned)
        background, border = kBannerColors[kBad if is_bad else kCaution]
        self.setStyleSheet(
            f"#checksBanner {{ background: {background};"
            f" border: 1px solid {border}; border-radius: 4px; }}"
        )
        sentence = " · ".join(
            verdictSentence(kind, verdict) for kind, verdict in warned
        )
        self.label.setText(sentence)
        self.label.setToolTip(sentence)
        self.btn_override.setVisible(
            model_check is not None and model_check.level == kBad and not is_overridden
        )
        self.show()


class ChecksCorner(QWidget):
    """The warning of the checks and the verdict of the tuning, where the
    gains are read. Made for the corner of a QTabWidget, which sizes it by
    its hint."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.banner = ChecksBanner()
        self.tuning_badge = CheckBadge(kTuning, prefix="Tuning check: ")
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 4, 0)
        layout.addWidget(self.banner, stretch=1)
        layout.addWidget(self.tuning_badge)
        self.setLayout(layout)

    def showChecks(self, model_check, tuning_check, is_overridden):
        self.banner.showChecks(model_check, tuning_check, is_overridden)
        self.tuning_badge.showCheck(tuning_check)
        self.updateGeometry()

    def sizeHint(self):
        hint = super().sizeHint()
        width = hint.width()
        if not self.banner.isHidden():
            width += self.banner.label.sizeHint().width()
        tabs = self.parentWidget()
        if tabs is not None and hasattr(tabs, "tabBar"):
            # A wider corner would squeeze the tabs
            width = min(width, tabs.width() - tabs.tabBar().sizeHint().width() - 8)
        return QSize(max(width, hint.width()), hint.height())
