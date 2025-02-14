# Import PySide6 / PyQt6 modules
from agepy.interactive.util import import_qt_binding
qt_binding, QtWidgets, QtCore, QtGui = import_qt_binding()
# Import ui modules
from .phex_dialog import PhexDialog
from .phem_dialog import Ui_PhemDialog
from .fit_setup import FitSetupDialog

__all__ = ["PhexDialog", "PhemDialog", "FitSetupDialog"]


class PhemDialog(QtWidgets.QDialog, Ui_PhemDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        if parent is not None:
            parent_rect = parent.geometry()
            self.move(parent_rect.topRight())

    def get_input(self):
        return (self.vpp_input.value(), self.Jpp_input.value())