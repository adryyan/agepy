"""Utility functions for gui.

"""
import warnings



def import_qt_binding():
    """Get the Qt binding that is available.

    Returns
    -------
    str
        The Qt binding that is available. Either "PySide6" or "PyQt6".

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
