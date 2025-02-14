# Import PySide6 / PyQt6 modules
from agepy.interactive.util import import_qt_binding
qt_binding, QtWidgets, QtCore, QtGui = import_qt_binding()


class PhexDialog(QtWidgets.QDialog):
    def __init__(self, parent, label, title="Assign Peak"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(400, 224)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.setFont(font)
        self.grid_layout = QtWidgets.QGridLayout(self)

        self.inputs = {}
        row = 0
        for key, value in label.items():
            group = QtWidgets.QGroupBox(key, parent=self)
            sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
            sizePolicy.setHorizontalStretch(0)
            sizePolicy.setVerticalStretch(0)
            sizePolicy.setHeightForWidth(group.sizePolicy().hasHeightForWidth())
            group.setSizePolicy(sizePolicy)
            layout = QtWidgets.QHBoxLayout(group)

            if isinstance(value, list):
                input_field = QtWidgets.QComboBox(parent=group)
                input_field.addItems(value)
                input_field.setEditable(True)
                input_field.lineEdit().setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                input_field.lineEdit().setReadOnly(True)
            elif isinstance(value, int):
                input_field = QtWidgets.QSpinBox(parent=group)
                input_field.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                input_field.setValue(value)
            else:
                raise ValueError("Unsupported value type in label dictionary")

            input_field.setObjectName(f"{key}_input")
            layout.addWidget(input_field)
            self.grid_layout.addWidget(group, row // 2, row % 2, 1, 1)
            self.inputs[key] = input_field
            row += 1

        self.button_box = QtWidgets.QDialogButtonBox(parent=self)
        self.button_box.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.button_box.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.Cancel | QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.grid_layout.addWidget(self.button_box, row // 2 + 1, 0, 1, 2)

        self.button_box.accepted.connect(self.accept)  # type: ignore
        self.button_box.rejected.connect(self.reject)  # type: ignore
        QtCore.QMetaObject.connectSlotsByName(self)

        if parent is not None:
            parent_rect = parent.geometry()
            self.move(parent_rect.topRight())

    def get_input(self):
        result = {}
        for key, input_field in self.inputs.items():
            if isinstance(input_field, QtWidgets.QComboBox):
                result[key] = input_field.currentText()
            elif isinstance(input_field, QtWidgets.QSpinBox):
                result[key] = input_field.value()
        return result
