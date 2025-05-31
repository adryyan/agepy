from __future__ import annotations

try:
    from PySide6 import QtGui

except ImportError:
    try:
        from PyQt6 import QtGui

    except ImportError as e:
        errmsg = "No compatible Qt bindings found. Install PySide6 or PyQt6."
        raise ImportError(errmsg) from e

from importlib.resources import path as ilrpath

from agepy.interactive import MainWindow
from agepy import ageplot

# Import modules for type hinting
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .scan import Scan
    from numpy.typing import ArrayLike


class SpectrumViewer(MainWindow):
    def __init__(self, scan: Scan, bins: int | ArrayLike) -> None:
        # Set up the main window
        super().__init__(title="Spectrum Viewer")
        self.add_plot()
        self.add_toolbar()
        self.add_forward_backward_action(self.plot_previous, self.plot_next)

        # Add actions for the calculation options
        self.calc_options = {}
        self.add_action_calc_option("qeff")
        self.add_action_calc_option("bkg")
        self.add_action_calc_option("calib")

        with ilrpath("agepy.interactive.icons", "errorbar.svg") as ipath:
            icon = QtGui.QIcon(str(ipath))

        self.add_action_calc_option("montecarlo", icon=icon)

        # Store references to the bins and the scan
        self.bins = bins
        self.scan = scan

        # Remember current step
        self.step = 0

        # Store current spectrum
        self.y, self.yerr, self.xe = None, None, None

        # Plot the first step
        self.plot()

    def add_action_calc_option(
        self, text: str, icon: QtGui.QIcon | None = None
    ) -> None:
        # Get the actions
        actions = self.toolbar.actions()

        # Create the action
        if icon is not None:
            action = QtGui.QAction(icon, text, self)

        else:
            action = QtGui.QAction(text, self)

        action.setCheckable(True)
        action.setChecked(False)

        # Connect the actions to the callback and add to toolbar
        action.triggered.connect(self.plot)
        self.toolbar.insertAction(actions[-1], action)
        self.calc_options[text] = action

    def plot(self) -> None:
        # Recalculate the spectrum
        if self.calc_options["montecarlo"].isChecked:
            uncertainties = "montecarlo"

        else:
            uncertainties = "poisson"

        self.y, self.yerr, self.xe = self.scan.spectrum_at(
            self.step,
            bins=self.bins,
            qeff=self.calc_options["qeff"].isChecked,
            bkg=self.calc_options["bkg"].isChecked,
            calib=self.calc_options["calib"].isChecked,
            uncertainties=uncertainties,
        )

        # Plot the spectrum
        with ageplot.context(["age", "interactive"]):
            # Clear the axes
            self.ax.clear()

            # Plot the spectrum
            self.ax.stairs(self.y, self.xe, color=ageplot.colors[0])

            # Plot the uncertainties
            self.ax.stairs(
                self.y + self.yerr,
                self.xe,
                baseline=self.y - self.yerr,
                color=ageplot.colors[0],
                alpha=0.5,
                fill=True,
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
