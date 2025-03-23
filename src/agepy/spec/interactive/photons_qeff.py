from __future__ import annotations
import warnings

# Import PySide6 / PyQt6 modules
try:
    from PySide6 import QtWidgets, QtCore, QtGui

    qt_binding = "PySide6"

except ImportError:
    warnings.warn("PySide6 not found, trying PyQt6. Some features may not work.")

    try:
        from PyQt6 import QtWidgets, QtCore, QtGui

        qt_binding = "PyQt6"

    except ImportError:
        raise ImportError("No compatible Qt bindings found.")

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

# Import internal modules
from agepy.spec.interactive.photons_scan import SpectrumViewer

# Import modules for type hinting
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agepy.spec.photons import Scan
    from matplotlib.backend_bases import MouseEvent
    from typing import Tuple
    from numpy.typing import NDArray


__all__ = ["EvalQEff"]


class EvalQEff(SpectrumViewer):
    """Evaluate a quantum efficiency measurement.

    """

    def __init__(self, scan: Scan, edges: np.ndarray, mc_samples: int) -> None:
        # Set up the main window
        super().__init__(scan, edges)

        # Add the fit action
        self.add_rect_selector(self.ax, self.on_select, hint="Select Peak")

        # Disable the qeff action
        self.calc_options[0] = False
        self.actions[0].setChecked(False)
        self.actions[0].setEnabled(False)

        # Disable the calib action
        self.calc_options[2] = False
        self.actions[2].setChecked(False)
        self.actions[2].setEnabled(False)

        # Force uncertainties to true
        self.calc_options[3] = True
        self.actions[3].setChecked(True)
        self.actions[3].setEnabled(False)

        # Set the number of Monte Carlo samples
        self.mc_samples = mc_samples

        # Plot the first step
        self.plot()

    def on_select(self, eclick: MouseEvent, erelease: MouseEvent):
        # Get the x selection
        xr = (eclick.xdata, erelease.xdata)

        # Select the data in the range
        in_range = np.argwhere((self.edges >= xr[0]) & (self.edges <= xr[1])).flatten()
        x = self.edges[in_range]
        y = self.y[in_range[:-1]]
        yerr = self.yerr[in_range[:-1]]

        # Fit the data
        n = np.stack((y, yerr**2), axis=-1)
        debug_fit = InteractiveFit(self, n, x, sig="Voigt", bkg="Constant")
        
        if debug_fit.exec():
            m = debug_fit.m

            # Get the fit parameters
            self.scan._py[self.step] = m.values["s"]
            self.scan._pyerr[self.step] = m.errors["s"]
            self.scan._px[self.step] = m.values["loc"]
            self.scan._pxerr[self.step] = m.errors["loc"]

        # Clear the selector
        self.selector.clear()

        # Plot the next step
        self.plot_next()


class InteractiveFit(QtWidgets.QDialog):

    def __init__(self,
        parent: QtWidgets.QWidget,
        n: NDArray,
        xe: NDArray,
        sig: str = "Gaussian",
        bkg: str = "None",
    ) -> None:
        # Initialize fit data
        self.n = n
        self.xe = xe

        # Set the x range
        self.xr = (xe[0], xe[-1])

        # Define starting values
        nsum = np.sum(n[:,0])
        self.s_start = nsum * 0.9
        self.s_limit = nsum * 1.1

        # Initialize the list of available models
        self._init_model_list()


        # Check if the signal model is available
        if sig not in self.sig_models.keys():
            raise ValueError(f"Signal model {sig} not found.")

        # Check if the background model is available
        if bkg not in self.bkg_models.keys():
            raise ValueError(f"Background model {bkg} not found.")

        # Initialize the parameters and limits
        self.params = {}

        # jit compile the numerical integration of the pdf

        @nb.njit(parallel=True, fastmath={"reassoc", "contract", "arcp"})
        def numint_cdf(_x, _pdf):
            y = np.empty_like(_x)
            for i in nb.prange(len(_x)):
                y[i] = np.trapz(_pdf[:i+1], x=_x[:i+1])
            return y

        self.numint_cdf = numint_cdf

        # Initialize the parent class
        super().__init__(parent)
        self.setWindowTitle("Quantum Efficiency Evaluation")
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
            QtWidgets.QSizePolicy.Policy.MinimumExpanding
        )

        # Create signal model selection widget
        self.signal_group = QtWidgets.QGroupBox("Signal Model")
        self.signal_group.setSizePolicy(size_policy)
        self.signal_layout = QtWidgets.QHBoxLayout(self.signal_group)
        self.sig_comp = QtWidgets.QComboBox()
        self.sig_comp.addItems(self.sig_models.keys())
        self.sig_comp.setCurrentIndex(list(self.sig_models.keys()).index(sig))
        self.sig_comp.currentIndexChanged.connect(self.prepare_fit)
        self.signal_layout.addWidget(self.sig_comp)
        self.layout.addWidget(self.signal_group, 1, 0, 1, 1)

        # Create background model selection widget
        self.background_group = QtWidgets.QGroupBox("Background Model")
        self.background_group.setSizePolicy(size_policy)
        self.background_layout = QtWidgets.QHBoxLayout(self.background_group)
        self.bkg_comp = QtWidgets.QComboBox()
        self.bkg_comp.addItems(self.bkg_models.keys())
        self.bkg_comp.setCurrentIndex(list(self.bkg_models.keys()).index(bkg))
        self.bkg_comp.currentIndexChanged.connect(self.prepare_fit)
        self.background_layout.addWidget(self.bkg_comp)
        self.layout.addWidget(self.background_group, 1, 1, 1, 1)

        # Create size policy
        size_policy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Minimum,
            QtWidgets.QSizePolicy.Policy.MinimumExpanding
        )

        # Create the button box
        self.button_group = QtWidgets.QGroupBox("Add Fit Result")
        self.button_group.setSizePolicy(size_policy)
        self.button_layout = QtWidgets.QHBoxLayout(self.button_group)
        self.button_box = QtWidgets.QDialogButtonBox(parent=self)
        self.button_box.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.button_box.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.Cancel | QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.button_box.accepted.connect(self.accept)  # type: ignore
        self.button_box.rejected.connect(self.reject)  # type: ignore
        QtCore.QMetaObject.connectSlotsByName(self)
        self.button_layout.addWidget(self.button_box)
        self.layout.addWidget(self.button_group, 1, 2, 1, 1, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)

        # Create the initial fit widget
        self.prepare_fit()

    def update_fit_widget(self, widget) -> None:
        # Remove the old fit widget
        self.layout.removeWidget(widget)
        # Set the new fit widget
        self.fit_widget = widget
        self.layout.addWidget(self.fit_widget, 0, 0, 1, 3)

    def prepare_fit(self) -> None:
        # Get the selected signal and background models
        sig = self.sig_comp.currentText()
        bkg = self.bkg_comp.currentText()

        # Remember current parameters and limits
        params_prev = self.params.copy()

        # Get the first signal model and parameters
        sig_integral, sig_params, sig_limits = self.sig_models[sig]()

        if bkg == "None":
            # Update the parameters and limits
            self.params = sig_params
            limits = sig_limits

            # Set the model

            def integral(x, *args):
                return sig_integral(x, args)

        else:
            # Get the background model and parameters
            bkg_integral, bkg_params, bkg_limits = self.bkg_models[bkg]()
            
            # Combine the parameters and limits
            self.params = {**sig_params, **bkg_params}
            limits = {**sig_limits, **bkg_limits}

            idx = len(sig_params)

            def integral(x, *args):
                return sig_integral(x, args[:idx]) + bkg_integral(x, args[idx:])

        # Keep previous parameters and limits if possible
        for par in params_prev:
            if par in self.params:
                self.params[par] = self.m.values[par]

        # Update the cost function
        c = cost.ExtendedBinnedNLL(self.n, self.xe, integral)

        # Update the Minuit object
        self.m = Minuit(c, *list(self.params.values()), name=list(self.params.keys()))
        
        # Set the limits
        for par, lim in limits.items():
            self.m.limits[par] = lim

        # Update the visualization
        fit_widget = make_widget(self.m, self.m._visualize(None), {}, False, False)

        # Perform the fit
        fit_widget.fit_button.click()

        # Update the layout
        self.update_fit_widget(fit_widget)

    def _init_model_list(self) -> None:
        self.sig_models = {
            "Gaussian": self.gaussian,
            "Q-Gaussian": self.qgaussian,
            "Voigt": self.voigt,
            "Cruijff": self.cruijff,
            "CrystalBall": self.crystalball,
            "CrystalBallEx": self.crystalball_ex,
        }
        self.bkg_models = {
            "None": None,
            "Constant": self.constant,
            "Exponential": self.exponential,
            "Bernstein1d": lambda: self.bernstein(1),
            "Bernstein2d": lambda: self.bernstein(2),
            "Bernstein3d": lambda: self.bernstein(3),
            "Bernstein4d": lambda: self.bernstein(4),
        }

    def gaussian(self) -> Tuple[callable, dict, dict]:
        dx = self.xr[1] - self.xr[0]
        params = {
            "s": self.s_start, "loc": 0.5 * (self.xr[0] + self.xr[1]),
            "scale": 0.05 * dx
        }
        limits = {
            "s": (0, self.s_limit), "loc": self.xr,
            "scale": (0.0001 * dx, 0.5 * dx)
        }

        def integral(x, par):
            return par[0] * norm.cdf(x, *par[1:])

        return integral, params, limits

    def qgaussian(self) -> Tuple[callable, dict, dict]:
        dx = self.xr[1] - self.xr[0]
        params = {
            "s": self.s_start, "q": 2, "loc": 0.5 * (self.xr[0] + self.xr[1]),
            "scale": 0.05 * dx
        }
        limits = {
            "s": (0, self.s_limit), "q": (1, 3), "loc": self.xr,
            "scale": (0.0001 * dx, 0.5 * dx)
        }

        def integral(x, par):
            if par[1] < 1:
                par[1] = 1
                warnings.warn("q cannot be smaller than 1. Setting q=1.")
            if par[1] > 3:
                par[1] = 3
                warnings.warn("q cannot be larger than 3. Setting q=3.")
            return par[0] * qgaussian.cdf(x, *par[1:])

        return integral, params, limits

    def voigt(self) -> Tuple[callable, dict, dict]:
        dx = self.xr[1] - self.xr[0]
        params = {
            "s": self.s_start, "gamma": 0.1 * dx,
            "loc": 0.5 * (self.xr[0] + self.xr[1]), "scale": 0.05 * dx
        }
        limits = {
            "s": (0, self.s_limit), "gamma": (0.0001 * dx, 0.5 * dx),
            "loc": self.xr, "scale": (0.0001 * dx, 0.5 * dx)
        }

        _x = np.linspace(self.xr[0], self.xr[1], 1000)

        def integral(x, par):
            _cdf = self.numint_cdf(_x, voigt.pdf(_x, *par[1:]))
            return par[0] * np.interp(x, _x, _cdf)

        return integral, params, limits

    def cruijff(self) -> Tuple[callable, dict, dict]:
        dx = self.xr[1] - self.xr[0]
        params = {
            "s": self.s_start, "beta_left": 0.1, "beta_right": 0.1,
            "loc": 0.5 * (self.xr[0] + self.xr[1]),
            "scale_left": 0.05 * dx, "scale_right": 0.05 * dx
        }
        limits = {
            "s": (0, self.s_limit), "beta_left": (0, 1), "beta_right": (0, 1),
            "loc": self.xr, "scale_left": (0.0001 * dx, 0.5 * dx),
            "scale_right": (0.0001 * dx, 0.5 * dx)
        }

        _x = np.linspace(self.xr[0], self.xr[1], 1000)

        def integral(x, par):
            _cdf = self.numint_cdf(_x, cruijff.density(_x, *par[1:]))
            return par[0] / _cdf[-1] * np.interp(x, _x, _cdf)

        return integral, params, limits

    def crystalball(self) -> Tuple[callable, dict, dict]:
        dx = self.xr[1] - self.xr[0]
        params = {
            "s": self.s_start, "beta": 1, "m": 2,
            "loc": 0.5 * (self.xr[0] + self.xr[1]), "scale": 0.05 * dx
        }
        limits = {
            "s": (0, self.s_limit), "beta": (0, 5), "m": (1, 10),
            "loc": self.xr, "scale": (0.0001 * dx, 0.5 * dx)
        }

        def integral(x, par):
            return par[0] * crystalball.cdf(x, *par[1:])

        return integral, params, limits

    def crystalball_ex(self) -> Tuple[callable, dict, dict]:
        dx = self.xr[1] - self.xr[0]
        params = {
            "s": self.s_start, "beta_left": 1, "m_left": 2,
            "scale_left": 0.05 * dx, "beta_right": 1, "m_right": 2,
            "scale_right": 0.05 * dx, "loc": 0.5 * (self.xr[0] + self.xr[1])
        }
        limits = {
            "s": (0, self.s_limit), "beta_left": (0, 5), "m_left": (1, 10),
            "scale_left": (0.0001 * dx, 0.5 * dx), "beta_right": (0, 5),
            "m_right": (1, 10), "scale_right": (0.0001 * dx, 0.5 * dx),
            "loc": self.xr
        }

        def integral(x, par):
            return par[0] * crystalball_ex.cdf(x, *par[1:])

        return integral, params, limits

    def constant(self) -> Tuple[callable, dict, dict]:
        params = {"b": 1}
        limits = {"b": (0, None)}

        def integral(x, par):
            return par[0] * uniform.cdf(x, self.xr[0], self.xr[1] - self.xr[0])

        return integral, params, limits

    def exponential(self) -> Tuple[callable, dict, dict]:
        params = {"b": 1, "loc_expon": 0, "scale_expon": 1}
        limits = {
            "b": (0, None), "loc_expon": (-1, 0), "scale_expon": (-100, 100)
        }

        def integral(x, par):
            return par[0] * truncexpon.cdf(x, self.xr[0], self.xr[1], *par[1:])

        return integral, params, limits

    def bernstein(self, deg: int) -> Tuple[callable, dict, dict]:
        params = {f"b_{i}{deg}": 1 for i in range(deg+1)}
        limits = {f"b_{i}{deg}": (0, None) for i in range(deg+1)}

        def integral(x, args):
            return bernstein.integral(x, args, self.xr[0], self.xr[1])

        return integral, params, limits
