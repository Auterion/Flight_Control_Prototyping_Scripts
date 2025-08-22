from PyQt5.QtCore import QSortFilterProxyModel, Qt
from PyQt5.QtWidgets import QComboBox, QCompleter


class SearchableComboBox(QComboBox):
    def __init__(self, parent=None):
        super(SearchableComboBox, self).__init__(parent)

        self.setFocusPolicy(Qt.ClickFocus)
        self.setEditable(True)

        # prevent insertions into combobox
        self.setInsertPolicy(QComboBox.NoInsert)

        # filter model for matching items
        self.pFilterModel = QSortFilterProxyModel(self)
        self.pFilterModel.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.pFilterModel.setSourceModel(self.model())

        # completer that uses filter model
        self.completer = QCompleter(self.pFilterModel, self)
        self.completer.setCompletionMode(QCompleter.UnfilteredPopupCompletion)
        self.setCompleter(self.completer)

        # connect signals
        self.lineEdit().textEdited[str].connect(self.pFilterModel.setFilterFixedString)
        self.completer.activated.connect(self.on_completer_activated)

    def on_completer_activated(self, text):
        if text:
            index = self.findText(text)
            self.setCurrentIndex(index)
            self.activated[str].emit(self.itemText(index))

    def setModel(self, model):
        super(SearchableComboBox, self).setModel(model)
        self.pFilterModel.setSourceModel(model)
        self.completer.setModel(self.pFilterModel)

    def setModelColumn(self, column):
        self.completer.setCompletionColumn(column)
        self.pFilterModel.setFilterKeyColumn(column)
        super(SearchableComboBox, self).setModelColumn(column)
