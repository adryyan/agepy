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

from .qt_scan import SpectrumViewer
from agepy.spec.fit_models import (
    SumModel1d,
    FitModel1d,
    Gaussian,
    Voigt,
    Constant,
    Exponential,
)
from agepy.qt.util import block_signals
from agepy import ageplot

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.backend_bases import MouseEvent
    from numpy.typing import NDArray, ArrayLike
    from .qeff import QEffScan


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
        self.fit_action, self.selector = self.add_rect_selector(
            self.ax, self.on_select, text="Select Peak", use_icon=True
        )

        with block_signals(*self.calc_options.values()):
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

            with ageplot.context(["age", "qt"]):
                # Plot the fit results
                self.ax.plot(x, y, color=ageplot.colors[1])
                self.ax.fill_between(
                    x, y - yerr, y + yerr, color=ageplot.colors[1], alpha=0.5
                )

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

            # Store the fit result
            self.scan.fit[self.step] = res

        # Close pyplot figures
        plt.close("all")

        # Clear the selector
        self.selector.clear()

        # Plot the next step
        self.plot_next()


class InteractiveFit(QtWidgets.QDialog):
    sig_models = {
        "Gaussian": Gaussian,
        "Voigt": Voigt,
    }

    bkg_models = {
        "Constant": Constant,
        "Exponential": Exponential,
    }

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        n: NDArray,
        xe: NDArray,
        sig: str = "Voigt",
        bkg: str = "Constant",
    ) -> None:
        # Initialize fit data
        self.n = n
        self.xe = xe

        # Set the x range
        self.xr = (xe[0], xe[-1])

        self.sig = None
        self.bkg = None
        self.fit = None

        # Initialize the parent class
        super().__init__(parent)
        self.setWindowTitle("Quantum Efficiency Fit")
        self.resize(1280, 720)
        self.setMaximumSize(1280, 720)
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
        if self.fit is None:
            self.fit = SumModel1d()

        # Get the selected signal model
        sig = self.sig_comp.currentText()
        if self.sig is None or self.sig.name != sig:
            self.sig = self.sig_models[sig](self.xr)
            self.fit.add_model(self.sig, idx=0)

        # Get the selected background model
        bkg = self.bkg_comp.currentText()
        if self.bkg is None or self.bkg.name != bkg:
            self.bkg = self.bkg_models[bkg](self.xr)
            self.fit.add_model(self.bkg, idx=1)

        # Update the cost function
        c = cost.ExtendedBinnedNLL(self.n, self.xe, self.fit.integral)

        # Update the Minuit object
        self.m = Minuit(c, *list(self.fit.val), name=self.fit.par)

        # Set the limits
        for par, lim in self.fit.lim.items():
            self.m.limits[par] = lim

        # Update the visualization
        fit_widget = make_widget(
            self.m, self.m._visualize(None), {}, False, False
        )

        # Perform the fit
        fit_widget.fit_button.click()

        # Update the layout
        self.update_fit_widget(fit_widget)

    def fit_result(self) -> FitModel1d | None:
        if not self.m.valid:
            return None

        # Get fitted parameter values and uncertainties
        self.fit.val = np.array(self.m.values)
        self.fit.err = np.array(self.m.errors)

        # Get the chi2 / ndof
        self.sig.chi2 = self.m.fmin.reduced_chi2

        return self.sig
