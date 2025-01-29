from __future__ import annotations
from typing import TYPE_CHECKING
# Import importlib.resources for getting the icon paths
from importlib.resources import path as ilrpath
# Import PySide6 / PyQt6 modules
from . util import import_qt_binding
qt_binding, QtWidgets, QtCore, QtGui = import_qt_binding()
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
    from typing import Union, Sequence
    from matplotlib.axes import Axes

__all__ = []


class AGEDataViewer(QtWidgets.QMainWindow):
    """Minimal implementation of the AGE Data Viewer.
    Should be used as a base class for more complex viewers.

    """
    def __init__(self, width: int = 1200, height: int = 800, parent=None):
        super().__init__(parent)
        # Set up the PyQt window
        self.setWindowTitle("AGE Data Viewer")
        self.setGeometry(100, 100, width, height)
        self.main_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QtWidgets.QVBoxLayout(self.main_widget)
        # Initialize attributes
        self.canvas = None
        self.toolbar = None

    def add_plot(
        self,
        fig: Figure = None,
        ax: Union[Axes, Sequence[Axes]] = None,
        layout: QtWidgets.QLayout = None,
        width: int = 1200,
        height: int = 800
    ) -> None:
        # Draw with the agepy plotting style, but don't overwrite the
        # users rcParams
        with ageplot.context(["age", "dataviewer"]):
            # Create and add the canvas
            if fig is not None:
                self.canvas = FigureCanvasQTAgg(fig)
            else:
                self.canvas = FigureCanvasQTAgg(Figure())
            # Set fixed size for the canvas
            self.canvas.setFixedSize(width, height)
            if layout is None:
                self.layout.addWidget(self.canvas)
            else:
                layout.addWidget(self.canvas)
            # Create the axis
            if ax is not None:
                self.ax = ax
            else:
                self.ax = self.canvas.figure.add_subplot(111)
            # Draw the empty plot
            self.canvas.draw()

    def add_toolbar(self):
        # Add the toolbar
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.layout.addWidget(self.toolbar)

    def add_roi_action(self, callback: callable, hint: str = "Add ROI") -> None:
        # Add ROI button to toolbar
        with ilrpath("agepy.interactive.icons", "roi.svg") as ipath:
            roi = QtGui.QAction(QtGui.QIcon(str(ipath)), hint, self)
        roi.setCheckable(True)
        roi.triggered.connect(callback)
        actions = self.toolbar.actions()
        self.roi_button = self.toolbar.insertAction(actions[-1], roi)

    def add_rect_selector(
        self,
        ax: Axes,
        on_select: callable,
        interactive: bool = True,
        hint: str = "Add ROI"
    ) -> None:
        # Add the action
        self.add_roi_action(self.toggle_selector, hint=hint)
        # Add ROI selector
        self.selector = RectangleSelector(
            ax, on_select,
            useblit=True,
            button=[1],
            minspanx=5, minspany=5,
            spancoords="pixels",
            interactive=interactive,
            props={"linewidth": 0.83, "linestyle": "--", "fill": False},
            handle_props={"markersize": 0})
        self.selector.set_active(False)

    def toggle_selector(self):
        self.selector.set_active(not self.selector.active)

    def add_forward_backward_action(
        self,
        bw_callback: callable,
        fw_callback: callable
    ) -> None:
        actions = self.toolbar.actions()
        # Add backward step to toolbar
        with ilrpath("agepy.interactive.icons", "bw-step.svg") as ipath:
            bw = QtGui.QAction(QtGui.QIcon(str(ipath)), "Step Backward", self)
        bw.triggered.connect(bw_callback)
        self.bw = self.toolbar.insertAction(actions[-1], bw)
        # Add forward step to toolbar
        with ilrpath("agepy.interactive.icons", "fw-step.svg") as ipath:
            fw = QtGui.QAction(QtGui.QIcon(str(ipath)), "Step Forward", self)
        fw.triggered.connect(fw_callback)
        self.fw = self.toolbar.insertAction(actions[-1], fw)

    def add_lookup_action(self, callback: callable) -> None:
        actions = self.toolbar.actions()
        with ilrpath("agepy.interactive.icons", "search.svg") as ipath:
            lu = QtGui.QAction(QtGui.QIcon(str(ipath)), "Look Up", self)
        lu.triggered.connect(callback)
        self.lu = self.toolbar.insertAction(actions[-1], lu)


class AGEpp:
    def __init__(self, viewer: QtWidgets.QMainWindow, *args, **kwargs):
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])
        self.viewer = viewer(*args, **kwargs)

    def run(self):
        self.viewer.show()
        return self.app.exec()
