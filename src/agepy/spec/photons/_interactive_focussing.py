from __future__ import annotations

try:
    from PySide6 import QtWidgets, QtCore, QtGui

    qt_binding = "PySide6"

except ImportError as e:
    errmsg = "PySide6 required for interactive fitting."
    raise ImportError(errmsg) from e

try:
    from iminuit import Minuit, cost
    from iminuit.qtwidget import make_widget

except ImportError as e:
    errmsg = "iminuit required for fitting."
    raise ImportError(errmsg) from e

import numpy as np
from jacobi import propagate
import matplotlib.pyplot as plt

from ._interactive_scan import SpectrumViewer
from ._interactive_fit import Gaussian, Voigt
from agepy.interactive import _block_signals
from agepy import ageplot


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.backend_bases import MouseEvent
    from numpy.typing import NDArray, ArrayLike
    from .focussing import FocusScan
    from ._interactive_fit import FitModel


class EvalFocus(SpectrumViewer):
    def __init__(
        self,
        scan: FocusScan,
        bins: int | ArrayLike,
        sig: str,
    ) -> None:
        # Set up the main window
        super().__init__(scan, bins)

        # Add the fit action
        self.add_rect_selector(self.ax, self.on_select, hint="Select Peak")

        with _block_signals(*self.calc_options.values()):
            # Disable the calib action
            self.calc_options["calib"].setChecked(False)
            self.calc_options["calib"].setEnabled(False)

            # Disable the qeff action
            self.calc_options["qeff"].setChecked(False)
            self.calc_options["qeff"].setEnabled(False)

            # Disable the montecarlo option
            self.calc_options["montecarlo"].setChecked(False)
            self.calc_options["montecarlo"].setEnabled(False)

            # Disablel the background options
            self.calc_options["bkg"].setChecked(False)
            self.calc_options["bkg"].setEnabled(False)

        # Set the default signal and background models
        self.default_sig = sig

        # Plot the first step
        self.plot()

    def plot(self):
        # Plot the spectrum
        super().plot()

        # Plot the fit
        fit = self.scan.fit[self.step]

        if fit is not None:
            x = np.linspace(0, 1, 1000)
            y, yerr = fit(x)

            # Scale to the bin width
            dx = self.xe[1] - self.xe[0]
            y *= dx
            yerr *= dx

            with ageplot.context(["age", "interactive"]):
                # Plot the fit results
                self.ax.plot(x, y, color=ageplot.colors[1])
                self.ax.fill_between(
                    x, y - yerr, y + yerr, color=ageplot.colors[1], alpha=0.5
                )

    def on_select(self, eclick: MouseEvent, erelease: MouseEvent):
        # Get the x selection
        xr = (eclick.xdata, erelease.xdata)

        # Get the xy data
        x, y = self.scan.spectra[self.step].xy()

        # Select the data in the range
        in_range = np.argwhere((x >= xr[0]) & (x <= xr[1])).flatten()
        x = x[in_range]
        y = y[in_range]

        # Get x edges
        xe = self.xe[in_range]

        # Interactively fit the data
        debug_fit = InteractiveFit(self, x, y, xe, sig=self.default_sig)

        if debug_fit.exec():
            sig1, sig2, res = debug_fit.fit_result()

            # Store the fit result
            self.scan.fit[self.step].append(sig1)
            self.scan.fit[self.step].append(sig2)
            self.scan.chi2[self.step].append(res["chi2"])

        # Close pyplot figures
        plt.close("all")

        # Clear the selector
        self.selector.clear()

        # Plot the next step
        self.plot_next()


class InteractiveFit(QtWidgets.QDialog):
    sig_models = {
        "Gaussian": lambda xr: Gaussian(xr),
        "Voigt": lambda xr: Voigt(xr),
    }

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        x: NDArray,
        y: NDArray,
        xe: NDArray,
        sig: str = "Voigt",
    ) -> None:
        # Initialize fit data
        self.x = x
        self.y = y

        # Get the bin edges and centers
        self.xe = xe
        self.xc = (xe[1:] + x[:-1]) * 0.5

        # Set the x range
        self.xr = (xe[0], xe[-1])

        # Define starting values and limits
        nsum = len(x)
        self.s_start = nsum * 0.9
        self.s_limit = nsum * 1.1

        self.sig1 = None
        self.sig2 = None

        # Initialize the parameters
        self.params = {}

        # Initialize the parent class
        super().__init__(parent)
        self.setWindowTitle("Photon Spectrum Fit")
        self.resize(1280, 720)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.setFont(font)

        self.layout = QtWidgets.QGridLayout(self)

        # Create dummy plot
        self.fit_widget = QtWidgets.QWidget()
        self.layout.addWidget(self.fit_widget, 0, 0, 1, 2)

        # Create size policy
        size_policy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.MinimumExpanding,
            QtWidgets.QSizePolicy.Policy.MinimumExpanding,
        )

        # Create signal model selection widget
        self.signal_group = QtWidgets.QGroupBox("Signal Model")
        self.signal_group.setSizePolicy(size_policy)
        self.signal_layout = QtWidgets.QHBoxLayout(self.signal_group)

        # Create ComboBox for the first signal component
        self.sig_comp1 = QtWidgets.QComboBox()
        self.sig_comp1.addItems(self.sig_models.keys())
        self.sig_comp1.setCurrentIndex(list(self.sig_models.keys()).index(sig))
        self.sig_comp1.currentIndexChanged.connect(self.prepare_fit)
        self.signal_layout.addWidget(self.sig_comp1)

        # Create ComboBox for the second signal component
        self.sig_comp2 = QtWidgets.QComboBox()
        self.sig_comp2.addItems(list(self.sig_models.keys()) + ["None"])
        self.sig_comp2.setCurrentIndex(len(self.sig_models))
        self.sig_comp2.currentIndexChanged.connect(self.prepare_fit)
        self.signal_layout.addWidget(self.sig_comp2)

        self.layout.addWidget(self.signal_group, 1, 0)

        # Create size policy
        size_policy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Minimum,
            QtWidgets.QSizePolicy.Policy.MinimumExpanding,
        )

        # Create the button box
        self.button_group = QtWidgets.QGroupBox("Add Fit Result")
        self.button_group.setSizePolicy(size_policy)
        self.button_layout = QtWidgets.QHBoxLayout(self.button_group)
        self.button_box = QtWidgets.QDialogButtonBox(parent=self)
        self.button_box.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.button_box.setStandardButtons(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
            | QtWidgets.QDialogButtonBox.StandardButton.Ok
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        QtCore.QMetaObject.connectSlotsByName(self)
        self.button_layout.addWidget(self.button_box)
        self.layout.addWidget(
            self.button_group,
            1,
            1,
            alignment=QtCore.Qt.AlignmentFlag.AlignLeft,
        )

        # Set the column stretch factors
        self.layout.setColumnStretch(0, 1)  # Signal group column
        self.layout.setColumnStretch(1, 0)  # Button group column

        # Create the initial fit widget
        self.prepare_fit()

    def update_fit_widget(self, widget: QtWidgets.QWidget) -> None:
        # Remove the old fit widget
        self.layout.removeWidget(widget)
        # Set the new fit widget
        self.fit_widget = widget
        self.layout.addWidget(self.fit_widget, 0, 0, 1, 2)

    def prepare_fit(self) -> None:
        # Get the selected signal and background models
        sig1 = self.sig_comp1.currentText()
        self.sig1 = self.sig_models[sig1](self.xr)

        sig2 = self.sig_comp2.currentText()
        if sig2 != "None":
            self.sig2 = self.sig_models[sig2](self.xr)

        else:
            self.sig2 = None

        # Remember current parameters and limits
        params_prev = self.params.copy()

        # Get the first signal model, starting values and limits
        par1 = self.sig1.start_val(self.s_start)
        lim1 = self.sig1.limits(self.s_limit)

        # Number of parameters for sig1
        idx1 = len(par1)

        # Append a 1 to the parameter names
        self.params = {f"{k}1": v for k, v in par1.items()}
        limits = {f"{k}1": v for k, v in lim1.items()}

        idx2 = 0

        if self.sig2 is not None:
            # Get the model and parameters of the second signal component
            par2 = self.sig2.start_val(self.s_start)
            lim2 = self.sig2.limits(self.s_limit)

            # Append a 2 to the parameter names
            par2 = {f"{k}2": v for k, v in par2.items()}
            lim2 = {f"{k}2": v for k, v in lim2.items()}

            # Combine the parameters and limits
            self.params = {**self.params, **par2}
            limits = {**limits, **lim2}

            # Shift the loc parameters
            self.params["loc1"] = 0.4 * (self.xr[1] - self.xr[0]) + self.xr[0]
            self.params["loc2"] = 0.6 * (self.xr[1] - self.xr[0]) + self.xr[0]

            # Index where the parameters of sig2 end
            idx2 = len(par2)

        def pdf(x, args):
            # Signal 1
            y = self.sig1.pdf(x, args[:idx1])

            # Signal 2
            if idx2 > 0:
                y += self.sig2.pdf(x, args[idx1 : idx1 + idx2])

            return y

        self.pdf = pdf

        def density(xy, *args):
            # Translate and rotated data
            x = self.transform(args[-3:])

            # Counts
            n = args[0] + args[idx1]

            return n, pdf(x, args)

        def plot(args):
            x = self.transform(args[-3:])

            # Histogram the data
            hist = np.histogram(x, bins=self.xe)[0]

            plt.errorbar(self.xc, hist, yerr=np.sqrt(hist), fmt="ok")

            # Get fit values
            xf = np.linspace(self.xr[0], self.xr[1], 1000)
            yf = pdf(xf, args)

            yf *= self.xe[1] - self.xe[0]

            plt.fill_between(x, yf)

        # Add the translation and rotation parameters
        self.params["theta"] = 0
        limits["theta"] = (-5, 5)

        self.params["dx"] = 0
        self.params["dy"] = 0
        limits["dx"] = (-0.01, 0.01)
        limits["dy"] = (-0.01, 0.01)

        # Keep previous parameters and limits if possible
        for par in params_prev:
            if par in self.params:
                self.params[par] = self.m.values[par]

        # Update the cost function
        c = cost.ExtendedUnbinnedNLL((self.x, self.y), density)

        # Update the Minuit object
        self.m = Minuit(
            c, *list(self.params.values()), name=list(self.params.keys())
        )

        # Set the limits
        for par, lim in limits.items():
            self.m.limits[par] = lim

        # Update the fit widget
        fit_widget = make_widget(self.m, plot, {}, False, False)

        # Perform the fit
        fit_widget.fit_button.click()

        # Update the layout
        self.update_fit_widget(fit_widget)

    def transform(self, args):
        # Shift the x and y values to origin
        x, y = self.x - 0.5, self.y - 0.5

        # Allow the fit to find a better shift (dx, dy)
        x = x + args[1]
        y = y + args[2]

        # Rotate the x and y values
        cos_theta = np.cos(-np.deg2rad(args[0]))
        sin_theta = np.sin(-np.deg2rad(args[0]))

        return (cos_theta * x - sin_theta * y) + 0.5

    def fit_result(
        self,
    ) -> tuple[FitModel | None, FitModel | None, dict | None]:
        if not self.m.valid:
            return None, None, None

        # Get the translation and rotation
        res = {
            "theta": (
                float(self.m.values["theta"]),
                float(self.m.errors["theta"]),
            ),
            "dx": (float(self.m.values["dx"]), float(self.m.errors["dx"])),
            "dy": (float(self.m.values["dy"]), float(self.m.errors["dy"])),
        }

        # Get the covariance matrix
        cov = np.array(self.m.covariance)

        # Get the fit parameter values
        values = np.array(self.m.values)

        # Calculate chi2
        x = self.transform(values[-3:])
        hist = np.histogram(x, bins=self.xe)[0]

        fit, err = propagate(lambda args: self.pdf(self.xc, args), values, cov)
        err = np.sqrt(np.diag(err))

        res["chi2"] = np.sum((hist - fit) ** 2 / err**2) / self.m.ndof

        # Append 1 to the parameter names
        par1 = [f"{k}1" for k in self.sig1.par]

        # Get fitted parameter values and uncertainties
        self.sig1.val = np.array(self.m.values[par1])
        self.sig1.err = np.array(self.m.errors[par1])

        # Get the covariance matrix for sig1
        self.sig1.cov = cov[: len(par1), : len(par1)]

        if self.sig2 is None:
            return self.sig1, None, res

        # Append 2 to the parameter names
        par2 = [f"{k}2" for k in self.sig2.par]

        # Get fitted parameter values and uncertainties
        self.sig2.val = np.array(self.m.values[par2])
        self.sig2.err = np.array(self.m.errors[par2])

        # Get the covariance matrix for sig2
        idx1 = len(par1)
        idx2 = idx1 + len(par2)
        self.sig2.cov = cov[idx1:idx2, idx1:idx2]

        return self.sig1, self.sig2, res
