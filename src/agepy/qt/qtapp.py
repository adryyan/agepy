from __future__ import annotations

try:
    from PySide6 import QtWidgets

except ImportError:
    try:
        from PyQt6 import QtWidgets

    except ImportError as e:
        raise ImportError("No compatible Qt bindings found.") from e

__all__ = []


def get_qtapp() -> QtWidgets.QApplication:
    # Get the current application instance
    app = QtWidgets.QApplication.instance()

    # Create a new application if none exists
    if app is None:
        app = QtWidgets.QApplication([])

    return app
