from __future__ import annotations
import warnings

# Import PySide6 / PyQt6 modules
try:
    from PySide6 import QtWidgets, QtCore, QtGui

    qt_binding = "PySide6"

except ImportError:
    raise ImportError("PySide6 required for interactive fitting.")

# Import the modules for the fitting
try:
    from iminuit import Minuit, cost
    from iminuit.qtwidget import make_widget
    from numba_stats import (
        bernstein,
        norm,
        truncexpon,
        uniform,
        voigt,
        cruijff,
        crystalball,
        crystalball_ex,
        qgaussian
    )

except ImportError:
    raise ImportError("iminuit and numba_stats required for fitting.")

import numpy as np
import numba as nb
import matplotlib.pyplot as plt

# Import internal modules
from agepy.interactive import _block_signals, MainWindow
from agepy import ageplot

# Import modules for type hinting
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agepy.spec.photons import Scan
    from matplotlib.backend_bases import MouseEvent
    from typing import Tuple
    from numpy.typing import NDArray


__all__ = ["EvalRot"]


class EvalRot(MainWindow):
    """Evaluate a quantum efficiency measurement.

    """

    def __init__(self, scan: Scan, intial_guess=0) -> None:
        self.scan = scan
        self.step = 0
        self.rot = intial_guess

        # Get the first detector image
        self.fig, ax = self.scan.spectra[self.step].det_image(
            num=42, figsize=(6, 6)
        )

        # Set up the main window
        super().__init__(title="Find Optimal Rotation", width=720, height=720)

        # Define size policy
        size_policy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.MinimumExpanding,
            QtWidgets.QSizePolicy.Policy.MinimumExpanding
        )

        # Set up a layout for the plot
        self.plot_widget = QtWidgets.QWidget(self)
        self.plot_widget.setSizePolicy(size_policy)
        self.plot_layout = QtWidgets.QVBoxLayout(self.plot_widget)
        self.layout.addWidget(self.plot_widget)

        # Set up the canvas
        self.add_plot(
            fig=self.fig, ax=ax, layout=self.plot_layout, width=720, height=720
        )

        # Add the toolbar
        self.add_toolbar()
        self.add_forward_backward_action(self.plot_previous, self.plot_next)

        # Add the fit action
        self.add_rect_selector(self.ax[0], self.on_select, hint="Select Line")

        # Plot the first step
        self.plot()

    def plot(self) -> None:
        # Plot the detector image
        with ageplot.context(["age", "interactive"]):
            # Plot the detector image
            self.scan.spectra[self.step].det_image(fig=self.fig, ax=self.ax)

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

    def on_select(self, eclick: MouseEvent, erelease: MouseEvent):
        # Get the x and y selection
        xr = (eclick.xdata, erelease.xdata)
        yr = (eclick.ydata, erelease.ydata)

        # Get the detector image
        x, y = self.scan.spectra[self.step].xy()

        # Select the data in the range
        mask = (x >= xr[0]) & (x <= xr[1]) & (y >= yr[0]) & (y <= yr[1])
        x = x[mask]
        y = y[mask]

        # Fit the data
        self.fit = InteractiveFit(self, x, y, self.rot)

        if self.fit.exec():
            # Get the fit parameters
            rot = self.fit.m.values["theta"]
            err = self.fit.m.errors["theta"]
            
            self.scan._rot[self.step] = rot
            self.scan._err[self.step] = err

        # Clear the selector
        self.selector.clear()

        # Plot the next step
        self.plot_next()


class InteractiveFit(QtWidgets.QDialog):

    def __init__(self,
        parent: QtWidgets.QWidget,
        x: NDArray,
        y: NDArray,
        start: float,
    ) -> None:
        # Initialize fit data
        self.x = x
        self.y = y

        # Get the x and y limits
        self.xlim = (x.min(), x.max())

        # Prepare starting values
        dx = self.xlim[1] - self.xlim[0]

        loc = 0.5 * (self.xlim[0] + self.xlim[1])
        gamma = 0.1 * dx
        theta = start

        # Initialize the parent class
        super().__init__(parent)
        self.setWindowTitle("Photon Spectrum Fit")
        self.resize(1280, 720)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.setFont(font)

        self.layout = QtWidgets.QVBoxLayout(self)

        def pdf(xy, gamma, loc, scale, theta):
            x, y = xy - 0.5
            cos_theta = np.cos(-theta * np.pi / 180)
            sin_theta = np.sin(-theta * np.pi / 180)
            x = cos_theta * x - sin_theta * y

            return voigt.pdf(x + 0.5, gamma, loc, scale)

        # Create the cost function
        c = cost.UnbinnedNLL((x, y), pdf)

        # Create the minimizer
        self.m = Minuit(c, gamma=gamma, loc=loc, scale=0.001, theta=theta)
        self.m.limits["gamma"] = (0.0001 * dx, 0.5 * dx)
        self.m.limits["loc"] = (self.xlim[0], self.xlim[1])
        self.m.limits["scale"] = (0.0001, 0.1)
        self.m.limits["theta"] = (-180, 180)

        def plot(args):
            # Rotate the data
            x, y = self.x - 0.5, self.y - 0.5
            cos_theta = np.cos(-args[3] * np.pi / 180)
            sin_theta = np.sin(-args[3] * np.pi / 180)
            x = cos_theta * x - sin_theta * y

            # Histogram the data
            hist, edges = np.histogram(
                x + 0.5, bins=512, range=(0, 1), density=True
            )

            # Get the bin centers
            centers = 0.5 * (edges[:-1] + edges[1:])

            # Plot the histogram
            plt.errorbar(centers, hist, yerr=np.sqrt(hist), fmt="o", color="b")

            # Plot the fit
            _x = np.linspace(0, 1, 1024)
            plt.plot(_x, voigt.pdf(_x, *args[:-1]), color="k")

            plt.xlim(self.xlim)

        # Update the fit widget
        self.fit_widget = make_widget(self.m, plot, {}, False, False)

        # Perform fit
        self.fit_widget.fit_button.click()

        # Add the fit widget to the layout
        self.layout.addWidget(self.fit_widget)

        # Create button box
        fix_policy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Fixed
        )
        self.button_box = QtWidgets.QWidget()
        self.button_layout = QtWidgets.QHBoxLayout(self.button_box)
        self.button_layout.addStretch()
        self.buttons = QtWidgets.QDialogButtonBox(parent=self)
        self.buttons.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.buttons.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.Cancel|QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.buttons.accepted.connect(self.accept)  # type: ignore
        self.buttons.rejected.connect(self.reject)  # type: ignore
        self.buttons.setSizePolicy(fix_policy)
        self.button_layout.addWidget(self.buttons)
        self.layout.addWidget(self.button_box)
