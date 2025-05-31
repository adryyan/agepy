from __future__ import annotations
import warnings

try:
    from PySide6 import QtWidgets, QtCore, QtGui

    qt_binding = "PySide6"

except ImportError as e:
    errmsg = "PySide6 required for interactive fitting."
    raise ImportError(errmsg) from e

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
        qgaussian,
    )

except ImportError as e:
    errmsg = "iminuit and numba_stats required for fitting."
    raise ImportError(errmsg) from e

import numpy as np
import numba as nb
from jacobi import propagate
import matplotlib.pyplot as plt

from ._interactive_scan import SpectrumViewer
from agepy.interactive import _block_signals

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
        self.add_rect_selector(self.ax, self.on_select, hint="Select Peak")

        with _block_signals(*self.calc_options.values()):
            # Disable the calib action
            self.calc_options["calib"].setChecked(False)
            self.calc_options["calib"].setEnabled(False)

            # Disable the qeff action
            self.calc_options["qeff"].setChecked(False)
            self.calc_options["qeff"].setEnabled(False)

            # Activate and disable the montecarlo option
            self.calc_options["montecarlo"].setChecked(True)
            self.calc_options["montecarlo"].setEnabled(False)

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
            res = debug_fit.fit_results()

            # Get the fit parameters
            self.scan._py[self.step] = res.val[0]
            self.scan._pyerr[self.step] = res.err[0]
            self.scan._px[self.step] = res.val[1]
            self.scan._pxerr[self.step] = res.err[1]

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
        self.layout.addWidget(self.fit_widget, 0, 0, 1, 4)

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

        self.layout.addWidget(self.signal_group, 1, 0, 1, 2)

        # Create background model selection widget
        self.background_group = QtWidgets.QGroupBox("Background Model")
        self.background_group.setSizePolicy(size_policy)
        self.background_layout = QtWidgets.QHBoxLayout(self.background_group)
        self.bkg_comp = QtWidgets.QComboBox()
        self.bkg_comp.addItems(self.bkg_models.keys())
        self.bkg_comp.setCurrentIndex(list(self.bkg_models.keys()).index(bkg))
        self.bkg_comp.currentIndexChanged.connect(self.prepare_fit)
        self.background_layout.addWidget(self.bkg_comp)
        self.layout.addWidget(self.background_group, 1, 2, 1, 1)

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
            3,
            1,
            1,
            alignment=QtCore.Qt.AlignmentFlag.AlignLeft,
        )

        # Set the column stretch factors
        self.layout.setColumnStretch(0, 2)  # Signal group column
        self.layout.setColumnStretch(1, 2)  # Signal group column
        self.layout.setColumnStretch(2, 1)  # Background group column
        self.layout.setColumnStretch(3, 0)  # Button group column

        # Create the initial fit widget
        self.prepare_fit()

    def update_fit_widget(self, widget: QtWidgets.QWidget) -> None:
        # Remove the old fit widget
        self.layout.removeWidget(widget)
        # Set the new fit widget
        self.fit_widget = widget
        self.layout.addWidget(self.fit_widget, 0, 0, 1, 4)

    def prepare_fit(self) -> None:
        # Get the selected signal model
        sig = self.sig_comp1.currentText()
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
                return sig.cdf(x, args)

        else:
            # Get the background model and parameters
            self.bkg = self.bkg_models[bkg](self.xr)

            par = self.bkg.start_val()
            lim = self.bkg.limits()

            idx = len(self.params)

            # Combine the parameters and limits
            self.params = {**self.params, **par}
            limits = {**limits, **lim}

            def integral(x, *args):
                return sig.cdf(x, args[:idx]) + bkg.cdf(x, args[idx:])

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


class FitModel:
    def __init__(
        self,
        xr: tuple[float, float],
    ) -> None:
        self.xr = xr

        # Fit results
        self.val = None
        self.err = None
        self.cov = None

    def __call__(self, x: NDArray) -> tuple[NDArray, NDArray]:
        if self.cov is None:
            raise ValueError("Fit results are not available.")

        # Propagate the uncertainties
        y, yerr = propagate(lambda par: self.pdf(x, par), self.val, self.cov)

        return y, np.sqrt(np.diag(yerr))

    def start_val(self, n):
        if self.val is not None:
            return {k: v for k, v in zip(self.par, self.val)}

        else:
            return self._start_val(n)


class Gaussian(FitModel):
    name = "Gaussian"
    par = ["s", "loc", "scale"]

    @staticmethod
    def pdf(x, par):
        return par[0] * norm.pdf(x, *par[1:])

    @staticmethod
    def cdf(x, par):
        return par[0] * norm.cdf(x, *par[1:])

    def limits(self, n_max):
        dx = self.xr[1] - self.xr[0]

        return {
            "s": (0, n_max),
            "loc": self.xr,
            "scale": (0.0001 * dx, 0.5 * dx),
        }

    def _start_val(self, n):
        return {
            "s": n,
            "loc": 0.5 * (self.xr[0] + self.xr[1]),
            "scale": 0.05 * (self.xr[1] - self.xr[0]),
        }


class Voigt(FitModel):
    name = "Voigt"
    par = ["s", "gamma", "loc", "scale"]

    @staticmethod
    def pdf(x, par):
        return par[0] * voigt.pdf(x, *par[1:])

    def cdf(self, x, par):
        _x = np.linspace(self.xr[0], self.xr[1], 1000)
        return par[0] * num_eval_cdf(x, _x, voigt.pdf(_x, *par[1:]))

    def limits(self, n_max):
        dx = self.xr[1] - self.xr[0]

        return {
            "s": (0, n_max),
            "gamma": (0.00001 * dx, 0.1 * dx),
            "loc": self.xr,
            "scale": (0.0001 * dx, 0.5 * dx),
        }

    def _start_val(self, n):
        dx = self.xr[1] - self.xr[0]

        return {
            "s": n,
            "gamma": 0.01 * dx,
            "loc": 0.5 * (self.xr[0] + self.xr[1]),
            "scale": 0.05 * (self.xr[1] - self.xr[0]),
        }


class QGaussian(FitModel):
    name = "Q-Gaussian"
    par = ["s", "q", "loc", "scale"]

    @staticmethod
    def pdf(x, par):
        if par[1] < 1:
            par[1] = 1
            wrnmsg = "q cannot be smaller than 1. Setting q=1."
            warnings.warn(wrnmsg, stacklevel=1)

        if par[1] > 3:
            par[1] = 3
            wrnmsg = "q cannot be larger than 3. Setting q=3."
            warnings.warn(wrnmsg, stacklevel=1)

        return par[0] * qgaussian.pdf(x, *par[1:])

    def cdf(self, x, par):
        if par[1] < 1:
            par[1] = 1
            wrnmsg = "q cannot be smaller than 1. Setting q=1."
            warnings.warn(wrnmsg, stacklevel=1)

        if par[1] > 3:
            par[1] = 3
            wrnmsg = "q cannot be larger than 3. Setting q=3."
            warnings.warn(wrnmsg, stacklevel=1)

        return par[0] * qgaussian.cdf(x, *par[1:])

    def limits(self, n_max):
        dx = self.xr[1] - self.xr[0]

        return {
            "s": (0, n_max),
            "q": (1, 3),
            "loc": self.xr,
            "scale": (0.0001 * dx, 0.5 * dx),
        }

    def _start_val(self, n):
        return {
            "s": n,
            "q": 2,
            "loc": 0.5 * (self.xr[0] + self.xr[1]),
            "scale": 0.05 * (self.xr[1] - self.xr[0]),
        }


class Cruijff(FitModel):
    name = "Cruijff"
    par = ["s", "beta_left", "beta_right", "loc", "scale_left", "scale_right"]

    @staticmethod
    def pdf(x, par):
        return par[0] * cruijff.density(x, *par[1:])

    def cdf(self, x, par):
        _x = np.linspace(self.xr[0], self.xr[1], 1000)
        return par[0] * num_eval_cdf(x, _x, cruijff.density(_x, *par[1:]))

    def limits(self, n_max):
        dx = self.xr[1] - self.xr[0]

        return {
            "s": (0, n_max),
            "beta_left": (0, 1),
            "beta_right": (0, 1),
            "loc": self.xr,
            "scale_left": (0.0001 * dx, 0.5 * dx),
            "scale_right": (0.0001 * dx, 0.5 * dx),
        }

    def _start_val(self, n):
        dx = self.xr[1] - self.xr[0]

        return {
            "s": n,
            "beta_left": 0.1,
            "beta_right": 0.1,
            "loc": 0.5 * (self.xr[0] + self.xr[1]),
            "scale_left": 0.05 * dx,
            "scale_right": 0.05 * dx,
        }


class CrystalBall(FitModel):
    name = "CrystalBall"
    par = ["s", "beta", "m", "loc", "scale"]

    @staticmethod
    def pdf(x, par):
        return par[0] * crystalball.pdf(x, *par[1:])

    @staticmethod
    def cdf(x, par):
        return par[0] * crystalball.cdf(x, *par[1:])

    def limits(self, n_max):
        dx = self.xr[1] - self.xr[0]

        return {
            "s": (0, n_max),
            "beta": (0, 5),
            "m": (1, 10),
            "loc": self.xr,
            "scale": (0.0001 * dx, 0.5 * dx),
        }

    def _start_val(self, n):
        dx = self.xr[1] - self.xr[0]

        return {
            "s": n,
            "beta": 1,
            "m": 2,
            "loc": 0.5 * (self.xr[0] + self.xr[1]),
            "scale": 0.05 * dx,
        }


class CrystalBallEx(FitModel):
    name = "CrystalBallEx"
    par = [
        "s",
        "beta_left",
        "m_left",
        "scale_left",
        "beta_right",
        "m_right",
        "scale_right",
        "loc",
    ]

    @staticmethod
    def pdf(x, par):
        return par[0] * crystalball_ex.pdf(x, *par[1:])

    @staticmethod
    def cdf(x, par):
        return par[0] * crystalball_ex.cdf(x, *par[1:])

    def limits(self, n_max):
        dx = self.xr[1] - self.xr[0]

        return {
            "s": (0, n_max),
            "beta_left": (0, 5),
            "m_left": (1, 10),
            "scale_left": (0.0001 * dx, 0.5 * dx),
            "beta_right": (0, 5),
            "m_right": (1, 10),
            "scale_right": (0.0001 * dx, 0.5 * dx),
            "loc": self.xr,
        }

    def _start_val(self, n):
        dx = self.xr[1] - self.xr[0]

        return {
            "s": n,
            "beta_left": 1,
            "m_left": 2,
            "scale_left": 0.05 * dx,
            "beta_right": 1,
            "m_right": 2,
            "scale_right": 0.05 * dx,
            "loc": 0.5 * (self.xr[0] + self.xr[1]),
        }


class Bernstein(FitModel):
    par = ["b_ij"]

    def __init__(
        self,
        deg: int,
        xr: tuple[float, float],
    ) -> None:
        super().__init__(xr)

        self.deg = deg
        self.par = [f"b_{i}{deg}" for i in range(deg + 1)]

    def pdf(self, x, par):
        return bernstein.density(x, par, *self.xr)

    def cdf(self, x, par):
        return bernstein.integral(x, par, *self.xr)

    def limits(self):
        return {f"b_{i}{self.deg}": (0, None) for i in range(self.deg + 1)}

    def start_val(self):
        return {f"b_{i}{self.deg}": 1 for i in range(self.deg + 1)}


class Constant(FitModel):
    par = ["b"]

    def pdf(self, x, par):
        return par[0] * uniform.pdf(x, self.xr[0], self.xr[1] - self.xr[0])

    def cdf(self, x, par):
        return par[0] * uniform.cdf(x, self.xr[0], self.xr[1] - self.xr[0])

    def limits(self):
        return {"b": (0, None)}

    def start_val(self):
        return {"b": 1}


class Exponential(FitModel):
    par = ["b", "loc_expon", "scale_expon"]

    def pdf(self, x, par):
        return par[0] * truncexpon.pdf(x, self.xr[0], self.xr[1], *par[1:])

    def cdf(self, x, par):
        return par[0] * truncexpon.cdf(x, self.xr[0], self.xr[1], *par[1:])

    def limits(self):
        return {
            "b": (0, None),
            "loc_expon": (-1, 0),
            "scale_expon": (-100, 100),
        }

    def start_val(self):
        return {"b": 1, "loc_expon": -0.5, "scale_expon": 1}


@nb.njit(parallel=True, fastmath={"reassoc", "contract", "arcp"})
def num_eval_cdf(x, _x, _pdf):
    _y = np.empty_like(_x)

    for i in nb.prange(len(_x)):
        _y[i] = np.trapz(_pdf[: i + 1], x=_x[: i + 1])

    return np.interp(x, _x, _y)
