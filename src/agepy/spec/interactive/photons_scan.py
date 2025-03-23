from __future__ import annotations
import warnings

# Import importlib.resources for getting the icon paths
from importlib.resources import path as ilrpath

# Import PySide6 / PyQt6 modules
try:
    from PySide6 import QtGui

    qt_binding = "PySide6"

except ImportError:
    warnings.warn("PySide6 not found, trying PyQt6. Some features may not work.")

    try:
        from PyQt6 import QtGui

        qt_binding = "PyQt6"

    except ImportError:
        raise ImportError("No compatible Qt bindings found.")

# Import internal modules
from agepy.interactive import MainWindow
from agepy import ageplot

# Import modules for type hinting
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agepy.spec.photons import Scan
    import numpy as np

__all__ = ["SpectrumViewer"]


class SpectrumViewer(MainWindow):
    """Show all spectra in a scan.

    """

    def __init__(self, scan: Scan, edges: np.ndarray) -> None:
        super().__init__(title="Spectrum Viewer")
        # Set up the main window
        self.add_plot()
        self.add_toolbar()
        self.add_forward_backward_action(self.plot_previous, self.plot_next)

        # Add actions for the calculation options
        self.calc_options = [False, False, False, False]
        self.actions = []
        self.add_action_calc_option("qeff")
        self.add_action_calc_option("bkg")
        self.add_action_calc_option("cal")
        with ilrpath("agepy.interactive.icons", "errorbar.svg") as ipath:
            icon = QtGui.QIcon(str(ipath))
        self.add_action_calc_option("Show Uncertainties", icon=icon)


        # Set the options
        self.mc_samples = 10000
        # Remember the edges and the scan
        self.edges = edges
        self.scan = scan
        # Intialize the data
        self.x = None
        self.y = None
        self.yerr = None
        # Remember current step
        self.step = 0
        # Plot the first step
        self.plot()

    def add_action_calc_option(self,
        text: str,
        icon: QtGui.QIcon = None
    ) -> None:
        # Get the action index
        index = len(self.actions)

        # Get the actions
        actions = self.toolbar.actions()

        # Create the action
        if icon is not None:
            action = QtGui.QAction(icon, text, self)

        else:
            action = QtGui.QAction(text, self)

        action.setCheckable(True)
        action.setChecked(self.calc_options[index])

        # Connect the actions to the callback and add to toolbar
        action.triggered.connect(lambda: self.set_calc_option(index))
        self.toolbar.insertAction(actions[-1], action)
        self.actions.append(action)

    def plot(self) -> None:
        # Get the calculation options
        qeff, bkg, calib, uncertainties = self.calc_options

        # Recalculate the spectrum
        if uncertainties:
            error_prop = "montecarlo"

        else:
            error_prop = "none"

        self.y, self.yerr = self.scan.spectrum_at(
            self.step, self.edges, qeff=qeff, bkg=bkg, calib=calib,
            err_prop=error_prop, mc_samples=self.mc_samples
        )

        if not uncertainties:
            self.yerr = None

        # Plot the spectrum
        with ageplot.context(["age", "interactive"]):
            # Clear the axes
            self.ax.clear()

            # Plot the spectrum
            self.ax.stairs(self.y, self.edges, color=ageplot.colors[0])

            # Plot the uncertainties
            if self.yerr is not None:
                self.ax.stairs(
                    self.y + self.yerr, self.edges, baseline=self.y - self.yerr,
                    color=ageplot.colors[0], alpha=0.5
                )

            # Refresh the canvas
            self.canvas.draw_idle()

    def plot_previous(self) -> None:
        if self.step - 1 < 0:
            return

        self.step -= 1
        self.plot()

    def plot_next(self) -> None:
        if self.step + 1 >= len(self.scan.steps):
            return

        self.step += 1
        self.plot()

    def set_calc_option(self, index: int) -> None:
        self.calc_options[index] = self.actions[index].isChecked()

        self.plot()
