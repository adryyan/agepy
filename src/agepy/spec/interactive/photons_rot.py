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
    from numba_stats import voigt

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

    def __init__(self,
        scan: Scan,
        angle: float,
        offset: Tuple[float, float]
    ) -> None:
        self.scan = scan
        self.step = 0

        # Set the starting values for the fit
        self.start_values = {
            "theta": np.radians(angle),
            "dx": offset[0],
            "dy": offset[1],
        }

        # Set the limits for the fit
        self.limits = {
            "theta": (-np.pi, np.pi),
            "dx": (-0.01, 0.01),
            "dy": (-0.01, 0.01),
        }

        # Cost function
        self.cost = None

        # Get the first detector image
        self.fig, ax = self.scan.spectra[self.step].det_image(
            num=42, figsize=(6, 6)
        )

        # Set up the main window
        super().__init__(title="Find Optimal Rotation", width=1080, height=1080)

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
            fig=self.fig, ax=ax, layout=self.plot_layout, width=1080, height=1080
        )

        # Add the toolbar
        self.add_toolbar()
        self.add_forward_backward_action(self.plot_previous, self.plot_next)

        # Add the selector
        self.add_rect_selector(self.ax[0], self.on_select, hint="Select Line")

        # Add the fit button
        # Get the actions
        actions = self.toolbar.actions()

        # Create the action
        self.fit_button = QtGui.QAction("Simul. Fit", self)

        # Connect the actions to the callback and add to toolbar
        self.fit_button.triggered.connect(self.fit)
        self.toolbar.insertAction(actions[-1], self.fit_button)

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

        # Get parameter names
        gamma = f"gamma{self.step}"
        loc = f"loc{self.step}"
        scale = f"scale{self.step}"
        par_names = [gamma, loc, scale, "theta", "dx", "dy"]

        # Add the starting values limits
        self.start_values[gamma] = 0.1 * (xr[1] - xr[0])
        self.limits[gamma] = (0.0001 * (xr[1] - xr[0]), 0.5 * (xr[1] - xr[0]))
        self.start_values[loc] = 0.5 * (xr[0] + xr[1])
        self.limits[loc] = xr
        self.start_values[scale] = 0.001
        self.limits[scale] = (0.0001, 0.1)

        # Add the cost function
        if self.cost is None:
            self.cost = cost.UnbinnedNLL((x, y), pdf, name=par_names)

        else:
            self.cost += cost.UnbinnedNLL((x, y), pdf, name=par_names)

        # Clear the selector
        self.selector.clear()

        # Plot the next step
        self.plot_next()

    def fit(self):
        # Get the parameter names
        par_names = [par for par in self.start_values]

        # Create the Minuit object
        m = Minuit(self.cost, **self.start_values)

        # Set the limits
        for par in self.limits:
            m.limits[par] = self.limits[par]

        # Fix all parameters except the angle
        m.fixed[par_names] = True
        m.fixed["theta"] = False

        # Get the indices of the added steps
        steps = [par[5:] for par in par_names if par.startswith("gamma")]

        # Fit the lines individually
        for step in steps:
            # Set the parameters for the current step
            m.fixed[f"gamma{step}"] = False
            m.fixed[f"loc{step}"] = False
            m.fixed[f"scale{step}"] = False

            # Continue the minimization
            m.migrad()

            # Fix the parameters for the current step
            m.fixed[f"gamma{step}"] = True
            m.fixed[f"loc{step}"] = True
            m.fixed[f"scale{step}"] = True

        # Let all parameters float except the offset
        m.fixed[par_names] = False
        m.fixed["dx"] = True
        m.fixed["dy"] = True

        # Continue the minimization
        m.migrad()

        # Also let the offset float
        m.fixed[par_names] = False

        # Finish the minimization
        m.migrad()

        # Check if the fit converged
        if not m.valid:
            msg_box = QtWidgets.QMessageBox(self)
            msg_box.setWindowTitle("Fit Error")
            msg_box.setText("Fit did not converge.")
            msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            msg_box.exec()
            return

        # Get the fit results
        self.scan._angle = np.array([
            np.degrees(m.values["theta"]), np.degrees(m.errors["theta"])
        ])

        self.scan._offset = np.array([
            [m.values["dx"], m.errors["dx"]], [m.values["dy"], m.errors["dy"]]
        ])

        # Create a message box to display the results
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setWindowTitle("Fit Results")
        msg_box.setText(
            f"Fit Results:\n\n"
            f"Angle: {self.scan._angle[0]:.2f} ± {self.scan._angle[1]:.2f} degrees\n"
            f"Offset X: {self.scan._offset[0, 0]:.4f} ± {self.scan._offset[0, 1]:.4f}\n"
            f"Offset Y: {self.scan._offset[1, 0]:.4f} ± {self.scan._offset[1, 1]:.4f}"
        )
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Information)
        msg_box.exec()


def pdf(xy, *args):
    # Shift the x and y values to origin
    x, y = xy - 0.5

    # Allow the fit to find a better shift (dx, dy)
    x = x + args[4]
    y = y + args[5]

    # Rotate the x and y values
    cos_theta = np.cos(-args[3])
    sin_theta = np.sin(-args[3])
    x = cos_theta * x - sin_theta * y

    # Evaluate Voigt at the new x values
    return voigt.pdf(x + 0.5, *args[:3])
