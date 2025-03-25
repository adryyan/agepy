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
    raise ImportError("iminuit and jax required for fitting.")

import numpy as np
import numba as nb
from jacobi import propagate
import pandas as pd
import matplotlib.pyplot as plt

# Import internal modules
from agepy.spec.interactive.assignment_dialog import AssignmentDialog
from agepy.spec.interactive.photons_scan import SpectrumViewer
from agepy import ageplot

# Import modules for type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence, Tuple, Dict, Union
    from matplotlib.backend_bases import MouseEvent
    from numpy.typing import NDArray
    from agepy.spec.photons import EnergyScan

__all__ = ["AssignPhem"]


class AssignPhem(SpectrumViewer):

    def __init__(self,
        scan: EnergyScan,
        edges: np.ndarray,
        reference: pd.DataFrame,
        phem_label: Dict[str, Union[Sequence[str], int]],
        phex_label: Dict[str, Union[Sequence[str], int]],
        calib_guess: Tuple[float, float],
    ) -> None:
        # Set the attributes
        self.reference = reference
        self.phem_label = phem_label
        self.phex_label = phex_label
        self.calib = calib_guess

        # Define x limits
        roi = scan.roi
        self.xlim = (roi[0][0], roi[0][1])

        # Prepare assignments
        if scan._phex_assignments is None:
            raise ValueError("Phex assignments are required.")

        self.phex = scan._phex_assignments.copy()

        if scan._phem_assignments is None:
            scan._phem_assignments = pd.DataFrame(
                columns=[*self.phem_label, *self.phex_label, "val", "err"]
            )

        # Set up the main window
        super().__init__(scan, edges)

        # Add the fit action
        self.add_rect_selector(
            self.ax, self.assign, interactive=False, hint="Assign Peak"
        )

        # Disable the calib action
        self.calc_options[2] = False
        self.actions[2].setChecked(False)
        self.actions[2].setEnabled(False)

        # Plot the first spectrum
        self.plot()

    def _parse_phex(self):
        phex = self.phex.iloc[self.step]
        phex_dict = {}
        phex_str = ""
        for key, val in phex.items():
            if key in ["E", "val", "err"]:
                continue
            phex_dict[key] = val
            phex_str += f"{key}={val}, "
        return phex, phex_dict, phex_str[:-2]

    def plot(self):
        # Prepare x -> wavelength conversion and vice versa
        a0, a1 = self.calib
        b1 = 1 / a1
        b0 = -a0 / a1
        wlim = (a1 * self.xlim[0] + a0, a1 * self.xlim[1] + a0)

        # Get the current phex assignment
        phex, phex_dict, phex_str = self._parse_phex()

        # Get the calculation options
        qeff, bkg, calib, uncertainties = self.calc_options

        # Recalculate the spectrum
        if uncertainties:
            error_prop = "montecarlo"

        else:
            error_prop = "none"

        self.y, self.yerr = self.scan.assigned_spectrum(
            phex_dict, self.edges, qeff=qeff, bkg=bkg, calib=calib,
            err_prop=error_prop, mc_samples=self.mc_samples
        )

        if not uncertainties:
            self.yerr = None

        with ageplot.context(["age", "interactive"]):
            # Clear the axes
            self.ax.clear()

            # Set the title
            self.ax.set_title(phex_str)

            # Plot the spectrum
            self.ax.stairs(self.y, self.edges, color=ageplot.colors[1])

            # Plot the uncertainties
            if self.yerr is not None:
                self.ax.stairs(
                    self.y + self.yerr, self.edges, baseline=self.y - self.yerr,
                    color=ageplot.colors[1], alpha=0.5
                )

            # Set the limits
            self.ax.set_xlim(*self.xlim)
            self.ax.set_ylim(0, np.max(self.y) * 1.1)

 
            # Plot assignments
            phem = self.scan._phem_assignments.copy()

            # Find the current phex assignment in the phem assignments
            for l, val in phex_dict.items():
                if phem.empty:
                    break

                if l not in phem:
                    continue

                phem.query(f"{l} == @val", inplace=True)

            # Plot the found assignments
            for i, row in phem.iterrows():
                # Get the fit results
                fit_val = row["val"]
                fit_err = row["err"]

                # Define x values
                x = np.linspace(*self.xlim, 1000)

                # Calculate the y values and their uncertainties
                y, yerr = propagate(
                    lambda par: par[0] * norm.pdf(x, *par[1:]), fit_val, fit_err**2
                )
                yerr = np.sqrt(np.diag(yerr))

                dx = self.edges[1] - self.edges[0]
                y *= dx
                yerr *= dx

                # Plot the fit results
                self.ax.plot(x, y, color=ageplot.colors[0])
                self.ax.fill_between(
                    x, y - yerr, y + yerr, color=ageplot.colors[0], alpha=0.5
                )

            # Plot reference
            ref = self.reference.query("E >= @wlim[0]").query("E <= @wlim[1]")

            # Find the current phex assignment in the reference
            for pl in self.phex_label:
                if ref.empty:
                    break

                if pl not in ref:
                    continue

                val = phex[pl]
                ref.query(f"{pl} == @val", inplace=True)

            # Plot the found reference
            for i, row in ref.iterrows():
                # Convert to detector position
                x = b1 * row["E"] + b0

                # Plot the reference
                self.ax.axvline(x, color="black", linestyle="--")

                # Create the text
                text = ""
                for label in self.phem_label:
                    text += f"{row[label]},"

                # Remove the last comma
                text = text[:-1]

                self.ax.text(
                    x + 0.0002, np.max(self.y), text, ha="left", va="top",
                    rotation=90
                )

            # Update the canvas
            self.canvas.draw_idle()

    def plot_next(self) -> None:
        if self.step + 1 >= len(self.phex):
            return

        self.step += 1
        self.plot()

    def assign(self, eclick: MouseEvent, erelease: MouseEvent):
        xr = (eclick.xdata, erelease.xdata)

        # Get the current phex assignment
        phex, phex_dict, phex_str = self._parse_phex()

        # Recalculate the spectrum with uncertainties
        if self.yerr is None:
            self.actions[3].setChecked(True)
            self.plot()
        
        # Prepare the data for the fit
        in_range = np.argwhere((self.edges >= xr[0]) & (self.edges <= xr[1])).flatten()
        xe = self.edges[in_range]
        y = self.y[in_range[:-1]]
        yerr = self.yerr[in_range[:-1]]
        n = np.stack((y, yerr**2), axis=-1)

        # Get the assignments
        dialog = AssignmentDialog(self, self.phem_label, title="Assign Peak 1")
        if dialog.exec():
            phem1 = dialog.get_input()

        else:
            return

        dialog = AssignmentDialog(self, self.phem_label, title="Assign Peak 2")
        if dialog.exec():
            phem2 = dialog.get_input()

            debug_fit = InteractiveFit(
                self, n, xe, sig=["Gaussian", "Gaussian"], bkg="Constant"
            )

        else:
            phem2 = None

            debug_fit = InteractiveFit(
                self, n, xe, sig="Gaussian", bkg="Constant"
            )

        # Fit the data
        if debug_fit.exec():
            m = debug_fit.m

            # Find the index where to save the assignment
            df = self.scan._phem_assignments.copy()
            for l, val in phem1.items():
                if df.empty:
                    break

                df.query(f"{l} == @val", inplace=True)

            for l, val in phex.items():
                phem1[l] = val
                df.query(f"{l} == @val", inplace=True)

            if df.empty:
                idx = self.scan._phem_assignments.index.max()
                if np.isnan(idx):
                    idx = 0

                else:
                    idx += 1

            else:
                idx = df.index[0]

            # Create the new row
            phem1["val"] = np.array(m.values["s1", "loc1", "scale1"])
            phem1["err"] = np.array(m.errors["s1", "loc1", "scale1"])

            # Save the assignment
            self.scan._phem_assignments.loc[idx] = phem1

            if phem2 is not None and  "loc2" in m.parameters:
                # Find the index where to save the assignment
                df = self.scan._phem_assignments.copy()
                for l, val in phem2.items():
                    if df.empty:
                        break

                    df.query(f"{l} == @val", inplace=True)

                for l, val in phex.items():
                    phem2[l] = val
                    df.query(f"{l} == @val", inplace=True)

                if df.empty:
                    idx = self.scan._phem_assignments.index.max() + 1
                else:
                    idx = df.index[0]

                # Create the new row
                phem2["val"] = np.array(m.values["s2", "loc2", "scale2"])
                phem2["err"] = np.array(m.errors["s2", "loc2", "scale2"])
            
                # Save the assignment
                self.scan._phem_assignments.loc[idx] = phem2

        # Plot the results
        self.plot()


class InteractiveFit(QtWidgets.QDialog):

    def __init__(self,
        parent: QtWidgets.QWidget,
        n: NDArray,
        xe: NDArray,
        sig: Union[str, Tuple[str, str]] = "Gaussian",
        bkg: str = "None",
    ) -> None:
        # Initialize fit data
        self.n = n
        self.xe = xe

        # Set the x range
        self.xr = (xe[0], xe[-1])

        # Define starting values and limits
        nsum = np.sum(n[:,0])
        self.s_start = nsum * 0.9
        self.s_limit = nsum * 1.1

        # Initialize the list of available models
        self._init_model_list()

        # Check the signal and background models
        if isinstance(sig, str):
            if sig not in self.sig_models.keys():
                raise ValueError(f"Signal model {sig} not found.")

            sig = [sig, "None"]

        else:
            if sig[0] not in self.sig_models.keys():
                raise ValueError(f"Signal model {sig[0]} not found.")

            if sig[1] not in list(self.sig_models.keys()) + ["None"]:
                raise ValueError(f"Signal model {sig[1]} not found.")

        if bkg not in self.bkg_models.keys():
            raise ValueError(f"Background model {bkg} not found.")

        # Initialize the parameters
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
        self.setWindowTitle("Photon Spectrum Fit")
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
            QtWidgets.QSizePolicy.Policy.MinimumExpanding
        )

        # Create signal model selection widget
        self.signal_group = QtWidgets.QGroupBox("Signal Model")
        self.signal_group.setSizePolicy(size_policy)
        self.signal_layout = QtWidgets.QHBoxLayout(self.signal_group)

        # Create ComboBox for the first signal component
        self.sig_comp1 = QtWidgets.QComboBox()
        self.sig_comp1.addItems(self.sig_models.keys())
        self.sig_comp1.setCurrentIndex(list(self.sig_models.keys()).index(sig[0]))
        self.sig_comp1.currentIndexChanged.connect(self.prepare_fit)
        self.signal_layout.addWidget(self.sig_comp1)

        # Create ComboBox for the second signal component
        self.sig_comp2 = QtWidgets.QComboBox()
        self.sig_comp2.addItems(list(self.sig_models.keys()) + ["None"])
        self.sig_comp2.setCurrentIndex((list(self.sig_models.keys()) + ["None"]).index(sig[1]))
        self.sig_comp2.currentIndexChanged.connect(self.prepare_fit)
        self.signal_layout.addWidget(self.sig_comp2)

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
        self.layout.addWidget(
            self.button_group, 1, 3, 1, 1,
            alignment=QtCore.Qt.AlignmentFlag.AlignLeft
        )

        # Create the initial fit widget
        self.prepare_fit()

    def update_fit_widget(self, widget) -> None:
        # Remove the old fit widget
        self.layout.removeWidget(widget)
        # Set the new fit widget
        self.fit_widget = widget
        self.layout.addWidget(self.fit_widget, 0, 0, 1, 4)

    def prepare_fit(self) -> None:
        # Get the selected signal and background models
        sig1 = self.sig_comp1.currentText()
        sig2 = self.sig_comp2.currentText()
        bkg = self.bkg_comp.currentText()

        # Remember current parameters and limits
        params_prev = self.params.copy()

        # Get the first signal model and parameters
        sig1_integral, sig1_params, sig1_limits = self.sig_models[sig1]()

        # Append a 1 to the parameter names
        sig1_params = {f"{k}1": v for k, v in sig1_params.items()}
        sig1_limits = {f"{k}1": v for k, v in sig1_limits.items()}

        # Initialize the signal and background models
        if bkg == "None" and sig2 == "None":
            # Update the parameters and limits
            self.params = sig1_params
            limits = sig1_limits

            def integral(x, *args):
                return sig1_integral(x, args)

        elif sig2 == "None":
            # Get the background model and parameters
            bkg_integral, bkg_params, bkg_limits = self.bkg_models[bkg]()
            
            # Combine the parameters and limits
            self.params = {**sig1_params, **bkg_params}
            limits = {**sig1_limits, **bkg_limits}

            idx = len(sig1_params)

            def integral(x, *args):
                return sig1_integral(x, args[:idx]) + bkg_integral(x, args[idx:])

        elif bkg == "None":
            # Get the model and parameters of the second signal component
            sig2_integral, sig2_params, sig2_limits = self.sig_models[sig2]()

            # Append a 2 to the parameter names
            sig2_params = {f"{k}2": v for k, v in sig2_params.items()}
            sig2_limits = {f"{k}2": v for k, v in sig2_limits.items()}
            
            # Combine the parameters and limits
            self.params = {**sig1_params, **sig2_params}
            limits = {**sig1_limits, **sig2_limits}

            idx = len(sig1_params)

            def integral(x, *args):
                return sig1_integral(x, args[:idx]) + sig2_integral(x, args[idx:])

        else:
            # Get the background model and parameters
            bkg_integral, bkg_params, bkg_limits = self.bkg_models[bkg]()

            # Get the model and parameters of the second signal component
            sig2_integral, sig2_params, sig2_limits = self.sig_models[sig2]()

            # Append a 2 to the parameter names
            sig2_params = {f"{k}2": v for k, v in sig2_params.items()}
            sig2_limits = {f"{k}2": v for k, v in sig2_limits.items()}

            # Combine the parameters and limits
            self.params = {**sig1_params, **sig2_params, **bkg_params}
            limits = {**sig1_limits, **sig2_limits, **bkg_limits}

            idx1 = len(sig1_params)
            idx2 = len(sig2_params) + idx1

            def integral(x, *args):
                return sig1_integral(x, args[:idx1]) + sig2_integral(x, args[idx1:idx2]) + bkg_integral(x, args[idx2:])

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
