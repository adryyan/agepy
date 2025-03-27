from __future__ import annotations
from typing import TYPE_CHECKING
import warnings
from contextlib import contextmanager

# Import importlib.resources for getting the icon paths
from importlib.resources import path as ilrpath

# Import PySide6 / PyQt6 modules
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

# Import matplotlib modules
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT
)
from matplotlib.widgets import RectangleSelector
from matplotlib.figure import Figure

# Internal agepy imports
from agepy import ageplot

# Import modules for type hinting
if TYPE_CHECKING:
    from typing import Union, Sequence, Literal
    from matplotlib.axes import Axes

__all__ = []


def get_qapp() -> QtWidgets.QApplication:
    # Get the current application instance
    app = QtWidgets.QApplication.instance()

    # Create a new application if none exists
    if app is None:
        app = QtWidgets.QApplication([])

    return app


class MainWindow(QtWidgets.QMainWindow):
    """Main window.

    """
    def __init__(self,
        width: int = 1280,
        height: int = 720,
        title: str = "AGE Interactive",
        layout: Literal["vertical", "horizontal"] = "vertical"
    ) -> None:
        super().__init__()
        # Set up the window
        self.setWindowTitle(title)
        self.setGeometry(100, 100, width, height)
        self.main_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.main_widget)

        # Set up the layout
        if layout == "vertical":
            self.layout = QtWidgets.QVBoxLayout(self.main_widget)
        elif layout == "horizontal":
            self.layout = QtWidgets.QHBoxLayout(self.main_widget)
        else:
            raise ValueError("Layout must be 'vertical' or 'horizontal'.")

        # Initialize attributes
        self.canvas = None
        self.toolbar = None

    def add_plot(self,
        fig: Figure = None,
        ax: Union[Axes, Sequence[Axes]] = None,
        layout: QtWidgets.QLayout = None,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        # Draw with the agepy plotting style, but don't overwrite the
        # users rcParams
        with ageplot.context(["age", "interactive"]):
            # Create and add the canvas
            if fig is not None:
                self.canvas = FigureCanvasQTAgg(fig)

            else:
                self.canvas = FigureCanvasQTAgg(Figure())

            # Set fixed size for the canvas
            self.canvas.setFixedSize(width, height)

            # Add the canvas to the layout
            if layout is None:
                self.layout.addWidget(self.canvas)

            else:
                layout.addWidget(self.canvas)

            # Create the axis
            if ax is not None:
                self.ax = ax

            else:
                self.ax = self.canvas.figure.add_subplot(111)

    def add_toolbar(self):
        # Check if a canvas exists
        if self.canvas is None:
            raise ValueError("No canvas to add toolbar to.")

        # Add the toolbar
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.layout.addWidget(self.toolbar)

    def add_action_select_data(self,
        callback: callable,
        hint: str = "Select Data"
    ) -> None:
        # Check if a toolbar exists
        if self.toolbar is None:
            raise ValueError("No toolbar to add actions to.")

        # Create the action
        with ilrpath("agepy.interactive.icons", "roi.svg") as ipath:
            action = QtGui.QAction(QtGui.QIcon(str(ipath)), hint, self)

        action.setCheckable(True)

        # Connect the action to the callback
        action.triggered.connect(callback)

        # Add ROI button to toolbar
        actions = self.toolbar.actions()
        self.button_select_data = self.toolbar.insertAction(actions[-1], action)

    def add_rect_selector(self,
        ax: Axes,
        on_select: callable,
        interactive: bool = True,
        hint: str = "Select Data"
    ) -> None:
        # Add the action
        self.add_action_select_data(self.toggle_selector, hint=hint)

        # Create data selector
        self.selector = RectangleSelector(
            ax, on_select,
            useblit=True,
            button=[1],
            minspanx=5, minspany=5,
            spancoords="pixels",
            interactive=interactive,
            props={"linewidth": 0.83, "linestyle": "--", "fill": False},
            handle_props={"markersize": 0}
        )

        # Deactivate selector
        self.selector.set_active(False)

    def toggle_selector(self):
        self.selector.set_active(not self.selector.active)

    def add_forward_backward_action(self,
        bw_callback: callable,
        fw_callback: callable
    ) -> None:
        # Check if a toolbar exists
        if self.toolbar is None:
            raise ValueError("No toolbar to add actions to.")
        
        # Get the actions
        actions = self.toolbar.actions()

        # Add backward step to toolbar
        with ilrpath("agepy.interactive.icons", "bw-step.svg") as ipath:
            bw = QtGui.QAction(QtGui.QIcon(str(ipath)), "Step Backward", self)

        # Connect the actions to the callback and add to toolbar
        bw.triggered.connect(bw_callback)
        self.bw = self.toolbar.insertAction(actions[-1], bw)

        # Add forward step to toolbar
        with ilrpath("agepy.interactive.icons", "fw-step.svg") as ipath:
            fw = QtGui.QAction(QtGui.QIcon(str(ipath)), "Step Forward", self)

        # Connect the actions to the callback and add to toolbar
        fw.triggered.connect(fw_callback)
        self.fw = self.toolbar.insertAction(actions[-1], fw)

    def add_lookup_action(self,
        callback: callable,
        hint: str = "Look Up"
    ) -> None:
        # Check if a toolbar exists
        if self.toolbar is None:
            raise ValueError("No toolbar to add actions to.")

        # Get the actions
        actions = self.toolbar.actions()

        # Add look up action to toolbar
        with ilrpath("agepy.interactive.icons", "search.svg") as ipath:
            lu = QtGui.QAction(QtGui.QIcon(str(ipath)), hint, self)

        # Connect the actions to the callback and add to toolbar
        lu.triggered.connect(callback)
        self.lu = self.toolbar.insertAction(actions[-1], lu)


@contextmanager
def _block_signals(*widgets):
    for w in widgets:
        w.blockSignals(True)

    try:
        yield

    finally:
        for w in widgets:
            w.blockSignals(False)
