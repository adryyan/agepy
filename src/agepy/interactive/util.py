"""Utility functions for gui.

"""
import warnings



def import_qt_binding():
    """Import the available Qt binding.

    """
    try:
        from PySide6 import QtWidgets, QtCore, QtGui

        qt_binding = "PySide6"
    except ImportError:
        warnings.warn("PySide6 not found, trying PyQt6. Some features may not work.")
        try:
            from PyQt6 import QtWidgets, QtCore, QtGui

            qt_binding = "PyQt6"
        except ImportError:
            raise ImportError("No compatible Qt bindings found.")

    return qt_binding, QtWidgets, QtCore, QtGui

def import_iminuit():
    """Import iminuit if installed.

    """
    try:
        from iminuit import Minuit, cost

    except ImportError:
        raise ImportError("iminuit not installed. Please install for fitting.")

    return Minuit, cost

def import_iminuit_interactive():
    """Import iminuit interactive if installed.

    """
    try:
        from iminuit.qtwidget import make_widget

    except ImportError:
        raise ImportError("iminuit>=2.31 not installed.")

    return make_widget
