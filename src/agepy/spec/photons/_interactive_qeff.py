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
import matplotlib.pyplot as plt

from ._interactive_scan import SpectrumViewer
from ._interactive_fit import (
    Gaussian,
    QGaussian,
    Voigt,
    Cruijff,
    CrystalBall,
    CrystalBallEx,
    Constant,
    Exponential,
    Bernstein,
)
from agepy.interactive import _block_signals

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.backend_bases import MouseEvent
    from numpy.typing import NDArray, ArrayLike
    from .qeff import QEffScan
    from ._interactive_fit import FitModel


class EvalQEff(SpectrumViewer):
    def __init__(
        self,
        scan: QEffScan,
        bins: int | ArrayLike,
        sig: str,
        bkg: str,
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

            # Activate and disable the montecarlo option
            self.calc_options["montecarlo"].setChecked(False)

            # Activate the other options
            self.calc_options["bkg"].setChecked(False)

        # Set the default signal and background models
        self.default_sig = sig
        self.default_bkg = bkg

        # Plot the first step
        self.plot()

    def on_select(self, eclick: MouseEvent, erelease: MouseEvent):
        # Get the x selection
        xr = (eclick.xdata, erelease.xdata)

        # Select the data in the range
        in_range = np.argwhere(
            (self.xe >= xr[0]) & (self.xe <= xr[1])
        ).flatten()

        # Prepare the data
        xe = self.xe[in_range]
        y = self.y[in_range[:-1]]
        yerr = self.yerr[in_range[:-1]]

        n = np.stack((y, yerr**2), axis=-1)

        # Interactively fit the data
        debug_fit = InteractiveFit(
            self, n, xe, sig=self.default_sig, bkg=self.default_bkg
        )

        if debug_fit.exec():
            res = debug_fit.fit_result()

            # Get the fit parameters
            self.scan._py[self.step] = res.value("s")
            self.scan._pyerr[self.step] = res.error("s")
            self.scan._px[self.step] = res.value("loc")
            self.scan._pxerr[self.step] = res.error("loc")

        # Close pyplot figures
        plt.close("all")

        # Clear the selector
        self.selector.clear()

        # Plot the next step
        self.plot_next()


class InteractiveFit(QtWidgets.QDialog):
    sig_models = {
        "Gaussian": lambda xr: Gaussian(xr),
        "Q-Gaussian": lambda xr: QGaussian(xr),
        "Voigt": lambda xr: Voigt(xr),
        "Cruijff": lambda xr: Cruijff(xr),
        "CrystalBall": lambda xr: CrystalBall(xr),
        "CrystalBallEx": lambda xr: CrystalBallEx(xr),
    }

    bkg_models = {
        "None": None,
        "Constant": lambda xr: Constant(xr),
        "Exponential": lambda xr: Exponential(xr),
        "Bernstein1d": lambda xr: Bernstein(1, xr),
        "Bernstein2d": lambda xr: Bernstein(2, xr),
        "Bernstein3d": lambda xr: Bernstein(3, xr),
        "Bernstein4d": lambda xr: Bernstein(4, xr),
    }

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        n: NDArray,
        xe: NDArray,
        sig: str = "Voigt",
        bkg: str = "None",
    ) -> None:
        # Initialize fit data
        self.n = n
        self.xe = xe

        # Set the x range
        self.xr = (xe[0], xe[-1])

        # Define starting values and limits
        nsum = np.sum(n[:, 0])
        self.s_start = nsum * 0.9
        self.s_limit = nsum * 1.1

        self.sig = None
        self.bkg = None

        # Initialize the parameters
        self.params = {}

        # Initialize the parent class
        super().__init__(parent)
        self.setWindowTitle("Quantum Efficiency Fit")
        self.resize(1280, 720)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.setFont(font)

        self.layout = QtWidgets.QGridLayout(self)

        # Create dummy plot
        self.fit_widget = QtWidgets.QWidget()
        self.layout.addWidget(self.fit_widget, 0, 0, 1, 3)

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
        self.sig_comp = QtWidgets.QComboBox()
        self.sig_comp.addItems(self.sig_models.keys())
        self.sig_comp.setCurrentIndex(list(self.sig_models.keys()).index(sig))
        self.sig_comp.currentIndexChanged.connect(self.prepare_fit)
        self.signal_layout.addWidget(self.sig_comp)

        self.layout.addWidget(self.signal_group, 1, 0)

        # Create background model selection widget
        self.background_group = QtWidgets.QGroupBox("Background Model")
        self.background_group.setSizePolicy(size_policy)
        self.background_layout = QtWidgets.QHBoxLayout(self.background_group)
        self.bkg_comp = QtWidgets.QComboBox()
        self.bkg_comp.addItems(self.bkg_models.keys())
        self.bkg_comp.setCurrentIndex(list(self.bkg_models.keys()).index(bkg))
        self.bkg_comp.currentIndexChanged.connect(self.prepare_fit)
        self.background_layout.addWidget(self.bkg_comp)
        self.layout.addWidget(self.background_group, 1, 1)

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
            2,
            alignment=QtCore.Qt.AlignmentFlag.AlignLeft,
        )

        # Set the column stretch factors
        self.layout.setColumnStretch(0, 1)  # Signal group column
        self.layout.setColumnStretch(1, 1)  # Background group column
        self.layout.setColumnStretch(2, 0)  # Button group column

        # Create the initial fit widget
        self.prepare_fit()

    def update_fit_widget(self, widget: QtWidgets.QWidget) -> None:
        # Remove the old fit widget
        self.layout.removeWidget(widget)
        # Set the new fit widget
        self.fit_widget = widget
        self.layout.addWidget(self.fit_widget, 0, 0, 1, 3)

    def prepare_fit(self) -> None:
        # Get the selected signal model
        sig = self.sig_comp.currentText()
        self.sig = self.sig_models[sig](self.xr)

        # Get the selected background model
        bkg = self.bkg_comp.currentText()

        # Remember current parameters and limits
        params_prev = self.params.copy()

        # Get the first signal model, starting values and limits
        self.params = self.sig.start_val(self.s_start)
        limits = self.sig.limits(self.s_limit)

        # Initialize the signal and background models
        if bkg == "None":

            def integral(x, *args):
                return self.sig.cdf(x, args)

        else:
            # Get the background model and parameters
            self.bkg = self.bkg_models[bkg](self.xr)

            par = self.bkg.start_val()
            lim = self.bkg.limits()

            i = len(self.params)

            # Combine the parameters and limits
            self.params = {**self.params, **par}
            limits = {**limits, **lim}

            def integral(x, *args):
                return self.sig.cdf(x, args[:i]) + self.bkg.cdf(x, args[i:])

        # Keep previous parameters and limits if possible
        for par in params_prev:
            if par in self.params:
                self.params[par] = self.m.values[par]

        # Update the cost function
        c = cost.ExtendedBinnedNLL(self.n, self.xe, integral)

        # Update the Minuit object
        self.m = Minuit(
            c, *list(self.params.values()), name=list(self.params.keys())
        )

        # Set the limits
        for par, lim in limits.items():
            self.m.limits[par] = lim

        # Update the visualization
        fit_widget = make_widget(
            self.m, self.m._visualize(None), {}, False, False
        )

        # Perform the fit
        fit_widget.fit_button.click()

        # Update the layout
        self.update_fit_widget(fit_widget)

    def fit_result(self) -> FitModel | None:
        if not self.m.valid:
            return None

        # Get the covariance matrix
        cov = np.array(self.m.covariance)

        # Get the parameter names
        par = self.sig.par

        # Get fitted parameter values and uncertainties
        self.sig.val = np.array(self.m.values[par])
        self.sig.err = np.array(self.m.errors[par])

        # Get the covariance matrix for sig1
        self.sig.cov = cov[: len(par), : len(par)]

        return self.sig
