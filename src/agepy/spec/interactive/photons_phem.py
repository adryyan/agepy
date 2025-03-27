from __future__ import annotations
import warnings

# Import PySide6 / PyQt6 modules
try:
    from PySide6 import QtWidgets, QtCore, QtGui

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
from jacobi import propagate
import pandas as pd
import matplotlib.pyplot as plt

# Import internal modules
from agepy.interactive import _block_signals
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
        label: Dict[str, Union[Sequence[str], int]],
        calib_guess: Tuple[float, float],
    ) -> None:
        # Set the attributes
        self.reference = reference
        self.label = label
        self.calib = calib_guess
        self.step = 0

        # Define x limits
        self.xlim = (scan.roi[0][0], scan.roi[0][1])

        # Prepare assignments
        if scan._phex_assignments is None:
            raise ValueError("Phex assignments are required.")

        self.phex = scan._phex_assignments.copy()

        # Create the phem assignments dataframe
        if scan._phem_assignments is None:
            col = [*self.label, *self.phex_to_dict(), "model", "fit"]
            scan._phem_assignments = pd.DataFrame(columns=col)

        # Set up the main window
        super().__init__(scan, edges)

        # Add the fit action
        self.add_rect_selector(
            self.ax, self.assign, interactive=False, hint="Assign Peaks"
        )

        with _block_signals(*self.actions):
            # Disable the calib action
            self.calc_options[2] = False
            self.actions[2].setChecked(False)
            self.actions[2].setEnabled(False)

            # Check the uncertainties, qeff and bkg actions
            self.calc_options[0] = True
            self.actions[0].setChecked(True)
            self.calc_options[1] = True
            self.actions[1].setChecked(True)
            self.calc_options[3] = True
            self.actions[3].setChecked(True)

        # Plot the spectrum with the adjusted settings
        self.plot()

    def phex_to_dict(self):
        # Get the current phex assignment
        phex = self.phex.iloc[self.step]

        # Create a dictionary
        phex_dict = {}

        for key, val in phex.items():
            if key in ["E", "val", "err"]:
                continue

            phex_dict[key] = val

        return phex_dict

    def find_phem(self, values: Dict[str, Union[str, int]]) -> pd.DataFrame:
        phem = self.scan._phem_assignments.copy()

        for key, val in values.items():
            if phem.empty:
                break

            if key not in phem:
                continue

            phem.query(f"{key} == @val", inplace=True)

        return phem

    def find_reference(self, values: Dict[str, Union[str, int]]) -> pd.DataFrame:
        ref = self.reference.copy()

        for key, val in values.items():
            if ref.empty:
                break

            if key not in ref:
                continue

            ref.query(f"{key} == @val", inplace=True)

        return ref

    def plot(self):
        # Prepare x -> wavelength conversion and vice versa
        a0, a1 = self.calib
        b1 = 1 / a1
        b0 = -a0 / a1
        wlim = (a1 * self.xlim[0] + a0, a1 * self.xlim[1] + a0)

        # Get the current phex assignment
        phex_dict = self.phex_to_dict()

        # Get the calculation options
        qeff, bkg, calib, uncertainties = self.calc_options

        # Recalculate the spectrum
        if uncertainties:
            error_prop = "montecarlo"

        else:
            error_prop = "none"

        # Create a progress dialog
        progress_dialog = QtWidgets.QProgressDialog(
            "Calculating...", None, 0, 0, self
        )
        progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.show()

        self.y, self.yerr = self.scan.assigned_spectrum(
            phex_dict, self.edges, n_std=1, normalize=True, qeff=qeff, bkg=bkg,
            calib=calib, err_prop=error_prop, mc_samples=self.mc_samples
        )

        # Close the progress dialog
        progress_dialog.close()

        if not uncertainties:
            self.yerr = None

        with ageplot.context(["age", "interactive"]):
            # Clear the axes
            self.ax.clear()

            # Set the title
            title = f"{phex_dict}"[1:-1].replace("'", "").replace(":", " =")
            self.ax.set_title(title)

            # Plot the spectrum
            self.ax.stairs(self.y, self.edges, color=ageplot.colors[1])

            # Plot the uncertainties
            if self.yerr is not None:
                self.ax.stairs(
                    self.y + self.yerr, self.edges, baseline=self.y - self.yerr,
                    color=ageplot.colors[1], fill=True, alpha=0.5
                )

            # Set the limits
            self.ax.set_xlim(*self.xlim)
            self.ax.set_ylim(0, np.max(self.y) * 1.1)

 
            # Plot assignments
            phem = self.find_phem(phex_dict)

            # Plot the found assignments
            for i, row in phem.iterrows():
                # Define x values
                x = np.linspace(*self.xlim, 1000)

                # Calculate the y values and their uncertainties
                y, yerr = row["fit"](x)

                # Scale to the bin width
                dx = self.edges[1] - self.edges[0]
                y *= dx
                yerr *= dx

                # Plot the fit results
                self.ax.plot(x, y, color=ageplot.colors[0])
                self.ax.fill_between(
                    x, y - yerr, y + yerr, color=ageplot.colors[0], alpha=0.5
                )

            # Plot reference
            ref = self.find_reference(phex_dict)

            # Select lines within the wavelength limits
            if not ref.empty:
                ref = ref.query("E >= @wlim[0]").query("E <= @wlim[1]")

            # Plot the found reference
            for i, row in ref.iterrows():
                # Convert to detector position
                x = b1 * row["E"] + b0

                # Plot the reference
                self.ax.axvline(x, color="black", linestyle="--")

                # Create the text
                text = ""
                for label in self.label:
                    text += f"{row[label]},"

                self.ax.text(
                    x + 0.0002, np.max(self.y), text[:-1], ha="left", va="top",
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
        phex_dict = self.phex_to_dict()

        # Recalculate the spectrum with uncertainties
        if self.yerr is None:
            self.actions[3].setChecked(True)
        
        # Prepare the data for the fit
        in_range = np.argwhere((
            self.edges >= xr[0]) & (self.edges <= xr[1])
        ).flatten()
        xe = self.edges[in_range]
        y = self.y[in_range[:-1]]
        yerr = self.yerr[in_range[:-1]]
        n = np.stack((y, yerr**2), axis=-1)

        # Get the assignment for the first peak
        dialog = AssignmentDialog(self, self.label, title="Assign Peak 1")
        if dialog.exec():
            # Get the user input
            phem1 = dialog.get_input()

            # Define starting values
            start_values = {}

            # Set the default signal model
            sig1_model = "Gaussian"
            sig2_model = "None"

            # Add the phex assignment
            phem1 = {**phem1, **phex_dict}

            # Find the index where to save the assignment
            df = self.find_phem(phem1)

            if df.empty:
                idx1 = self.scan._phem_assignments.index.max()
                if np.isnan(idx1):
                    idx1 = 0

                else:
                    idx1 += 1

            else:
                idx1 = df.index[0]

                # Get the model and starting values
                sig1_model = df.loc[idx1, "model"]
                start_values = df.loc[idx1, "fit"].start_val(1)
                start_values = {f"{k}1": v for k, v in start_values.items()}

        else:
            return

        # Get the assignment for the second peak
        dialog = AssignmentDialog(self, self.label, title="Assign Peak 2")
        if dialog.exec():
            # Get the user input
            phem2 = dialog.get_input()

            # Set the default signal model
            sig2_model = "Gaussian"

            # Add the phex assignment
            phem2 = {**phem2, **phex_dict}

            # Find the index where to save the assignment
            df = self.find_phem(phem2)

            if df.empty:
                idx2 = self.scan._phem_assignments.index.max()

                if np.isnan(idx2) or idx2 + 1 == idx1:
                    idx2 = idx1 + 1

                else:
                    idx2 += 1

            else:
                idx2 = df.index[0]

                # Get the model and starting values
                sig2_model = df.loc[idx2, "model"]
                sv2 = df.loc[idx2, "fit"].start_val(1)
                sv2 = {f"{k}2": v for k, v in sv2.items()}
                start_values = {**start_values, **sv2}

        else:
            phem2 = None

        debug_fit = InteractiveFit(
            self, n, xe, sig=[sig1_model, sig2_model], bkg="Constant",
            **start_values
        )

        # Fit the data
        if debug_fit.exec():
            # Get the fit results
            res1, res2 = debug_fit.fit_result()

            # Don't save the assignment if the fit failed
            if res1 is None:
                return

            # Create the new row
            phem1 = {**phem1, **res1}

            # Save the assignment
            self.scan._phem_assignments.loc[idx1] = phem1

            if phem2 is not None and  res2 is not None:
                # Create the new row
                phem2 = {**phem2, **res2}

                # Save the assignment
                self.scan._phem_assignments.loc[idx2] = phem2

        # Close pyplot figures
        plt.close("all")

        # Plot the results
        self.plot()


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

    def __init__(self,
        parent: QtWidgets.QWidget,
        n: NDArray,
        xe: NDArray,
        sig: Union[str, Tuple[str, str]] = "Gaussian",
        bkg: str = "None",
        **start_values
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
        self.start_values = start_values

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

        # Set signal and background models
        self.sig1 = None
        self.sig2 = None
        self.bkg = None

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
        self.sig_comp1.setCurrentIndex(
            list(self.sig_models.keys()).index(sig[0])
        )
        self.sig_comp1.currentIndexChanged.connect(self.prepare_fit)
        self.signal_layout.addWidget(self.sig_comp1)

        # Create ComboBox for the second signal component
        self.sig_comp2 = QtWidgets.QComboBox()
        self.sig_comp2.addItems(list(self.sig_models.keys()) + ["None"])
        self.sig_comp2.setCurrentIndex(
            (list(self.sig_models.keys()) + ["None"]).index(sig[1])
        )
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

        # Set the column stretch factors
        self.layout.setColumnStretch(0, 2)  # Signal group column
        self.layout.setColumnStretch(1, 2)  # Signal group column
        self.layout.setColumnStretch(2, 1)  # Background group column
        self.layout.setColumnStretch(3, 0)  # Button group column

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

        # Get the first signal model, starting values and limits
        sig1 = self.sig_models[sig1](self.xr)
        par1 = sig1.start_val(self.s_start)
        lim1 = sig1.limits(self.s_limit)

        # Append a 1 to the parameter names
        par1 = {f"{k}1": v for k, v in par1.items()}
        lim1 = {f"{k}1": v for k, v in lim1.items()}

        self.sig1 = sig1

        # Initialize the signal and background models
        if bkg == "None" and sig2 == "None":
            # Update the parameters and limits
            self.params = par1
            limits = lim1

            def integral(x, *args):
                return sig1.cdf(x, args)

        elif sig2 == "None":
            # Get the background model and parameters
            bkg = self.bkg_models[bkg](self.xr)
            par2 = bkg.start_val()
            lim2 = bkg.limits()

            self.bkg = bkg
            self.sig2 = None

            # Combine the parameters and limits
            self.params = {**par1, **par2}
            limits = {**lim1, **lim2}

            idx = len(par1)

            def integral(x, *args):
                return sig1.cdf(x, args[:idx]) + bkg.cdf(x, args[idx:])

        elif bkg == "None":
            # Get the model and parameters of the second signal component
            sig2 = self.sig_models[sig2](self.xr)
            par2 = sig2.start_val(self.s_start)
            lim2 = sig2.limits(self.s_limit)

            # Append a 2 to the parameter names
            par2 = {f"{k}2": v for k, v in par2.items()}
            lim2 = {f"{k}2": v for k, v in lim2.items()}

            self.sig2 = sig2
            self.bkg = None

            # Combine the parameters and limits
            self.params = {**par1, **par2}
            limits = {**lim1, **lim2}

            idx = len(par1)

            def integral(x, *args):
                return sig1.cdf(x, args[:idx]) + sig2.cdf(x, args[idx:])

        else:
            # Get the model and parameters of the second signal component
            sig2 = self.sig_models[sig2](self.xr)
            par2 = sig2.start_val(self.s_start)
            lim2 = sig2.limits(self.s_limit)

            # Append a 2 to the parameter names
            par2 = {f"{k}2": v for k, v in par2.items()}
            lim2 = {f"{k}2": v for k, v in lim2.items()}

            self.sig2 = sig2

            # Get the background model and parameters
            bkg = self.bkg_models[bkg](self.xr)
            par3 = bkg.start_val()
            lim3 = bkg.limits()

            self.bkg = bkg

            # Combine the parameters and limits
            self.params = {**par1, **par2, **par3}
            limits = {**lim1, **lim2, **lim3}

            idx1 = len(par1)
            idx2 = len(par2) + idx1

            def integral(x, *args):
                return sig1.cdf(x, args[:idx1]) + sig2.cdf(x, args[idx1:idx2]) \
                    + bkg.cdf(x, args[idx2:])

        # Overwrite with given starting values
        for par, val in self.start_values.items():
            if par in self.params:
                self.params[par] = val

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
        fit_widget = make_widget(self.m, self.m._visualize(None), {}, False, False)

        # Perform the fit
        fit_widget.fit_button.click()

        # Update the layout
        self.update_fit_widget(fit_widget)

    def fit_result(self):
        if not self.m.valid:
            return None, None

        # Get the covariance matrix
        cov = np.array(self.m.covariance)

        # Get the fit results for sig1
        sig1 = {}

        # Get the model name
        sig1["model"] = self.sig_comp1.currentText()

        # Append 1 to the parameter names
        par1 = [f"{k}1" for k in self.sig1.par]

        # Get fitted parameter values and uncertainties
        self.sig1.val = np.array(self.m.values[par1])
        self.sig1.err = np.array(self.m.errors[par1])

        # Get the covariance matrix for sig1
        self.sig1.cov = cov[:len(par1), :len(par1)]
        
        # Set the fit results for sig1
        sig1["fit"] = self.sig1

        if self.sig2 is None:
            return sig1, None

        # Get the fit results for sig2
        sig2 = {}

        # Get the model name
        sig2["model"] = self.sig_comp2.currentText()

        # Append 2 to the parameter names
        par2 = [f"{k}2" for k in self.sig2.par]

        # Get fitted parameter values and uncertainties
        self.sig2.val = np.array(self.m.values[par2])
        self.sig2.err = np.array(self.m.errors[par2])

        # Get the covariance matrix for sig2
        idx1 = len(par1)
        idx2 = idx1 + len(par2)
        self.sig2.cov = cov[idx1:idx2, idx1:idx2]
        
        # Set the fit results for sig2
        sig2["fit"] = self.sig2

        return sig1, sig2


class FitModel:

    def __init__(self,
        xr: Tuple[float, float],
    ) -> None:
        self.xr = xr

        # Fit results
        self.val = None
        self.err = None
        self.cov = None

    def __call__(self, x: NDArray) -> Tuple[NDArray, NDArray]:
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
            "s": (0, n_max), "loc": self.xr, "scale": (0.0001 * dx, 0.5 * dx)
        }

    def _start_val(self, n):
        return {
            "s": n, "loc": 0.5 * (self.xr[0] + self.xr[1]),
            "scale": 0.05 * (self.xr[1] - self.xr[0])
        }


class Voigt(FitModel):

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
            "s": (0, n_max), "gamma": (0.0001 * dx, 0.5 * dx),
            "loc": self.xr, "scale": (0.0001 * dx, 0.5 * dx)
        }

    def _start_val(self, n):
        dx = self.xr[1] - self.xr[0]

        return {
            "s": n, "gamma": 0.1 * dx, "loc": 0.5 * (self.xr[0] + self.xr[1]),
            "scale": 0.05 * (self.xr[1] - self.xr[0])
        }


class QGaussian(FitModel):

    par = ["s", "q", "loc", "scale"]

    @staticmethod
    def pdf(x, par):
        if par[1] < 1:
            par[1] = 1
            warnings.warn("q cannot be smaller than 1. Setting q=1.")

        if par[1] > 3:
            par[1] = 3
            warnings.warn("q cannot be larger than 3. Setting q=3.")

        return par[0] * qgaussian.pdf(x, *par[1:])

    def cdf(self, x, par):
        if par[1] < 1:
            par[1] = 1
            warnings.warn("q cannot be smaller than 1. Setting q=1.")

        if par[1] > 3:
            par[1] = 3
            warnings.warn("q cannot be larger than 3. Setting q=3.")

        return par[0] * qgaussian.cdf(x, *par[1:])

    def limits(self, n_max):
        dx = self.xr[1] - self.xr[0]

        return {
            "s": (0, n_max), "q": (1, 3), "loc": self.xr,
            "scale": (0.0001 * dx, 0.5 * dx)
        }

    def _start_val(self, n):
        return {
            "s": n, "q": 2, "loc": 0.5 * (self.xr[0] + self.xr[1]),
            "scale": 0.05 * (self.xr[1] - self.xr[0])
        }


class Cruijff(FitModel):

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
            "s": (0, n_max), "beta_left": (0, 1), "beta_right": (0, 1),
            "loc": self.xr, "scale_left": (0.0001 * dx, 0.5 * dx),
            "scale_right": (0.0001 * dx, 0.5 * dx)
        }

    def _start_val(self, n):
        dx = self.xr[1] - self.xr[0]

        return {
            "s": n, "beta_left": 0.1, "beta_right": 0.1,
            "loc": 0.5 * (self.xr[0] + self.xr[1]),
            "scale_left": 0.05 * dx, "scale_right": 0.05 * dx
        }


class CrystalBall(FitModel):

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
            "s": (0, n_max), "beta": (0, 5), "m": (1, 10),
            "loc": self.xr, "scale": (0.0001 * dx, 0.5 * dx)
        }

    def _start_val(self, n):
        dx = self.xr[1] - self.xr[0]

        return {
            "s": n, "beta": 1, "m": 2,
            "loc": 0.5 * (self.xr[0] + self.xr[1]), "scale": 0.05 * dx
        }


class CrystalBallEx(FitModel):

    par = [
        "s", "beta_left", "m_left", "scale_left", "beta_right", "m_right",
        "scale_right", "loc"
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
            "s": (0, n_max), "beta_left": (0, 5), "m_left": (1, 10),
            "scale_left": (0.0001 * dx, 0.5 * dx), "beta_right": (0, 5),
            "m_right": (1, 10), "scale_right": (0.0001 * dx, 0.5 * dx),
            "loc": self.xr
        }
    
    def _start_val(self, n):
        dx = self.xr[1] - self.xr[0]

        return {
            "s": n, "beta_left": 1, "m_left": 2, "scale_left": 0.05 * dx,
            "beta_right": 1, "m_right": 2, "scale_right": 0.05 * dx,
            "loc": 0.5 * (self.xr[0] + self.xr[1])
        }


class Bernstein(FitModel):

    par = ["b_ij"]

    def __init__(self,
        deg: int,
        xr: Tuple[float, float],
    ) -> None:
        super().__init__(xr)

        self.deg = deg
        self.par = [f"b_{i}{deg}" for i in range(deg+1)]

    def pdf(self, x, par):
        return bernstein.density(x, par, *self.xr)

    def cdf(self, x, par):
        return bernstein.integral(x, par, *self.xr)

    def limits(self):
        return {f"b_{i}{self.deg}": (0, None) for i in range(self.deg+1)}

    def start_val(self):
        return {f"b_{i}{self.deg}": 1 for i in range(self.deg+1)}


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
            "b": (0, None), "loc_expon": (-1, 0), "scale_expon": (-100, 100)
        }

    def start_val(self):
        return {"b": 1, "loc_expon": -0.5, "scale_expon": 1}


@nb.njit(parallel=True, fastmath={"reassoc", "contract", "arcp"})
def num_eval_cdf(x, _x, _pdf):
    _y = np.empty_like(_x)

    for i in nb.prange(len(_x)):
        _y[i] = np.trapz(_pdf[:i+1], x=_x[:i+1])

    return np.interp(x, _x, _y)
