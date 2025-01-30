# Import PySide6 / PyQt6 modules
from agepy.interactive.util import import_qt_binding
qt_binding, QtWidgets, QtCore, QtGui = import_qt_binding()
# Import ui modules
from .phex_dialog import Ui_PhexDialog

__all__ = ["PhexDialog"]


class PhexDialog(QtWidgets.QDialog, Ui_PhexDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.el_input.addItems(["B"])
        self.el_input.setEditable(True)
        self.el_input.lineEdit().setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.el_input.lineEdit().setReadOnly(True)
        self.br_input.addItems(["P", "Q", "R"])
        self.br_input.setEditable(True)
        self.br_input.lineEdit().setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.br_input.lineEdit().setReadOnly(True)
        if parent is not None:
            parent_rect = parent.geometry()
            self.move(parent_rect.topRight())

    def get_input(self):
        return (self.el_input.currentText(), self.vp_input.value(),
                self.Jp_input.value(), self.br_input.currentText())
