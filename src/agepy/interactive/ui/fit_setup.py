# Import PySide6 / PyQt6 modules
from agepy.interactive.util import import_qt_binding
qt_binding, QtWidgets, QtCore, QtGui = import_qt_binding()


class FitSetupDialog(QtWidgets.QDialog):
    def __init__(self,
        signal_entries,
        background_entries,
        max_signal=None,
        max_background=None,
        parent=None
    ) -> None:
        super().__init__(parent)
        self.signal_entries = signal_entries
        self.background_entries = background_entries
        self.max_signal = max_signal
        self.max_background = max_background

        self.setWindowTitle("Fit Setup")

        # Set size
        self.resize(500, 300)
        exp_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding, QtWidgets.QSizePolicy.Policy.MinimumExpanding)
        fix_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)

        # Set Font
        font = QtGui.QFont()
        font.setPointSize(14)
        self.setFont(font)

        # Create main layout
        main_layout = QtWidgets.QGridLayout(self)

        # Create labels
        self.signal_label = QtWidgets.QLabel("Signal Model")
        self.background_label = QtWidgets.QLabel("Background Model")

        # Create button box
        self.button_box = QtWidgets.QWidget()
        self.button_layout = QtWidgets.QHBoxLayout(self.button_box)
        self.button_layout.addStretch()
        self.buttons = QtWidgets.QDialogButtonBox(parent=self)
        self.buttons.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.buttons.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.Cancel|QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.buttons.setSizePolicy(fix_policy)
        self.button_layout.addWidget(self.buttons)

        # Create scroll areas
        self.signal_scroll_area = QtWidgets.QScrollArea()
        self.signal_scroll_area.setSizePolicy(exp_policy)
        self.background_scroll_area = QtWidgets.QScrollArea()
        self.background_scroll_area.setSizePolicy(exp_policy)

        # Create container widgets for scroll areas
        self.signal_container = QtWidgets.QWidget()
        self.background_container = QtWidgets.QWidget()

        # Create layouts for container widgets
        self.signal_layout = QtWidgets.QVBoxLayout(self.signal_container)
        self.background_layout = QtWidgets.QVBoxLayout(self.background_container)

        # Set container widgets to scroll areas
        self.signal_scroll_area.setWidget(self.signal_container)
        self.background_scroll_area.setWidget(self.background_container)

        # Set scroll areas to be scrollable
        self.signal_scroll_area.setWidgetResizable(True)
        self.background_scroll_area.setWidgetResizable(True)

        # Add widgets to main layout
        main_layout.addWidget(self.signal_label, 0, 0)
        main_layout.addWidget(self.background_label, 0, 1)
        main_layout.addWidget(self.signal_scroll_area, 1, 0)
        main_layout.addWidget(self.background_scroll_area, 1, 1)
        main_layout.addWidget(self.button_box, 2, 1)

        # Set titles for scroll areas
        self.signal_scroll_area.setWindowTitle("Signal Model")
        self.background_scroll_area.setWindowTitle("Background Model")

        # Connect buttons
        self.buttons.accepted.connect(self.accept) # type: ignore
        self.buttons.rejected.connect(self.reject) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(self)

        # Initialize combobox lists
        self.signal_comboboxes = []
        self.background_comboboxes = []

        # Add buttons at the bottom
        self.add_signal_button = QtWidgets.QPushButton("+")
        self.add_signal_button.clicked.connect(self.add_signal_combobox)
        self.signal_layout.addWidget(self.add_signal_button)
        self.signal_layout.addStretch()

        self.add_background_button = QtWidgets.QPushButton("+")
        self.add_background_button.clicked.connect(self.add_background_combobox)
        self.background_layout.addWidget(self.add_background_button)
        self.background_layout.addStretch()

        # Create initial comboboxes and add buttons
        self.add_signal_combobox()
        self.signal_comboboxes[0].setCurrentIndex(0)
        self.add_background_combobox()
        self.background_comboboxes[0].setCurrentIndex(1)
        self.background_comboboxes[0].setCurrentIndex(0)

    def add_signal_combobox(self):
        combobox = NoScrollQComboBox()
        combobox.addItems(self.signal_entries)
        combobox.currentTextChanged.connect(
            lambda text, cb=combobox: self.remove_signal_combobox(text, cb))
        self.signal_layout.insertWidget(self.signal_layout.count() - 2, combobox)
        self.signal_comboboxes.append(combobox)
        if len(self.signal_comboboxes) == self.max_signal:
            self.add_signal_button.setEnabled(False)

    def add_background_combobox(self):
        combobox = NoScrollQComboBox()
        combobox.addItems(self.background_entries)
        combobox.currentTextChanged.connect(
            lambda text, cb=combobox: self.remove_background_combobox(text, cb))
        self.background_layout.insertWidget(self.background_layout.count() - 2, combobox)
        self.background_comboboxes.append(combobox)
        if len(self.background_comboboxes) == self.max_background:
            self.add_background_button.setEnabled(False)

    def remove_signal_combobox(self, text, combobox):
        if text == "None":
            index = self.signal_layout.indexOf(combobox)
            self.signal_layout.takeAt(index).widget().deleteLater()
            self.signal_comboboxes.remove(combobox)
            if len(self.signal_comboboxes) < self.max_signal:
                self.add_signal_button.setEnabled(True)

    def remove_background_combobox(self, text, combobox):
        if text == "None":
            index = self.background_layout.indexOf(combobox)
            self.background_layout.takeAt(index).widget().deleteLater()
            self.background_comboboxes.remove(combobox)
            if len(self.background_comboboxes) < self.max_background:
                self.add_background_button.setEnabled(True)

    def get_selected_entries(self):
        signal_entries = [cb.currentText() for cb in self.signal_comboboxes if cb.currentText() != "None"]
        background_entries = [cb.currentText() for cb in self.background_comboboxes if cb.currentText() != "None"]
        return signal_entries, background_entries


class NoScrollQComboBox(QtWidgets.QComboBox):
    def __init__(self, *args, **kwargs):
        super(NoScrollQComboBox, self).__init__(*args, **kwargs)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def wheelEvent(self, event):
        if self.hasFocus():
            return QtWidgets.QComboBox.wheelEvent(self, event)
        else:
            return event.ignore()
