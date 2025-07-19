from __future__ import annotations

from importlib.resources import path as ilrpath

try:
    from PySide6 import QtWidgets, QtGui

except ImportError:
    try:
        from PyQt6 import QtWidgets, QtGui

    except ImportError as e:
        errmsg = "No compatible Qt bindings found."
        raise ImportError(errmsg) from e

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.widgets import RectangleSelector
from matplotlib.figure import Figure

from agepy import ageplot

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Union, Sequence, Literal
    from matplotlib.axes import Axes

__all__ = []


class MainWindow(QtWidgets.QMainWindow):
    """Main window."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        title: str = "AGE Main Window",
        layout: Literal["vertical", "horizontal"] = "vertical",
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
            errmsg = "Layout must be 'vertical' or 'horizontal'."
            raise ValueError(errmsg)

        # Initialize attributes
        self.canvas = None
        self.toolbar = None

    def add_plot(
        self,
        fig: Figure | None = None,
        ax: Union[Axes, Sequence[Axes]] | None = None,
        layout: QtWidgets.QLayout | None = None,
        width: int = 960,
        height: int = 720,
        tight_layout: bool = True,
    ) -> None:
        # Draw with the agepy plotting style, but don't overwrite the
        # users rcParams
        with ageplot.context(["age", "qt"]):
            # Create and add the canvas
            if fig is not None:
                self.canvas = FigureCanvasQTAgg(fig)
                self.fig = fig

            else:
                self.fig = Figure(tight_layout=tight_layout)
                self.canvas = FigureCanvasQTAgg(self.fig)

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
                self.ax = self.fig.add_subplot(111)

    def add_toolbar(self):
        # Check if a canvas exists
        if self.canvas is None:
            errmsg = "No canvas to add toolbar to."
            raise AttributeError(errmsg)

        # Add the toolbar
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.layout.addWidget(self.toolbar)

    def add_action(
        self,
        callback: callable,
        text: str,
        icon: QtGui.QIcon | None = None,
        checkable: bool = False,
    ) -> QtGui.QAction:
        if self.toolbar is None:
            errmsg = "Add toolbar before an action."
            raise AttributeError(errmsg)

        if icon is not None:
            action = QtGui.QAction(icon, text, self)

        else:
            action = QtGui.QAction(text, self)

        if checkable:
            action.setCheckable(True)
            action.setChecked(False)

        action.triggered.connect(callback)

        actions = self.toolbar.actions()
        self.toolbar.insertAction(actions[-1], action)

        return action

    def add_action_selector(
        self,
        callback: callable,
        text: str = "Rectangle Selector",
        use_icon: bool = False,
    ) -> QtGui.QAction:
        icon = None
        if use_icon:
            with ilrpath("agepy.qt.icons", "roi.svg") as ipath:
                icon = QtGui.QIcon(str(ipath))

        return self.add_action(callback, text, icon=icon, checkable=True)

    def add_rect_selector(
        self,
        ax: Axes,
        on_select: callable,
        interactive: bool = True,
        text: str = "Select Data",
        use_icon: bool = False,
    ) -> tuple[QtGui.QAction, RectangleSelector]:
        # Create data selector
        selector = RectangleSelector(
            ax,
            on_select,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=interactive,
            props={"linewidth": 0.83, "linestyle": "--", "fill": False},
            handle_props={"markersize": 0},
        )

        def toggle_selector():
            selector.set_active(not selector.active)

        # Add the action
        action = self.add_action_selector(
            toggle_selector,
            text=text,
            use_icon=use_icon,
        )

        # Deactivate selector
        selector.set_active(False)

        return action, selector

    def add_action_prev_next(
        self, prev_callback: callable, next_callback: callable
    ) -> None:
        with ilrpath("agepy.qt.icons", "bw-step.svg") as ipath:
            prev_icon = QtGui.QIcon(str(ipath))

        prev_action = self.add_action(
            prev_callback, "Previous Step", prev_icon
        )

        with ilrpath("agepy.qt.icons", "fw-step.svg") as ipath:
            next_icon = QtGui.QIcon(str(ipath))

        next_action = self.add_action(next_callback, "Next Step", next_icon)

        return prev_action, next_action

    def add_lookup_action(
        self, callback: callable, hint: str = "Look Up"
    ) -> None:
        # Check if a toolbar exists
        if self.toolbar is None:
            raise ValueError("No toolbar to add actions to.")

        # Get the actions
        actions = self.toolbar.actions()

        # Add look up action to toolbar
        with ilrpath("agepy.qt.icons", "search.svg") as ipath:
            lu = QtGui.QAction(QtGui.QIcon(str(ipath)), hint, self)

        # Connect the actions to the callback and add to toolbar
        lu.triggered.connect(callback)
        self.lu = self.toolbar.insertAction(actions[-1], lu)
