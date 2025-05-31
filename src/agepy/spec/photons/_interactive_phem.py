from __future__ import annotations

try:
    from PySide6 import QtWidgets, QtCore, QtGui

except ImportError as e:
    errmsg = "PySide6 required for interactive fitting."
    raise ImportError(errmsg) from e

# Import the modules for the fitting
try:
    from iminuit import Minuit, cost
    from iminuit.qtwidget import make_widget

except ImportError as e:
    errmsg = "iminuit required for fitting."
    raise ImportError(errmsg) from e

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agepy.interactive import _block_signals
from ._assignment_dialog import AssignmentDialog
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
from agepy import ageplot

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.backend_bases import MouseEvent
    from numpy.typing import NDArray, ArrayLike
    from .energy_scan import EnergyScan
    from ._interactive_fit import FitModel


class AssignPhem(SpectrumViewer):
    def __init__(
        self,
        scan: EnergyScan,
        reference: pd.DataFrame,
        calib: tuple[float, float],
        bins: int | ArrayLike,
    ) -> None:
        # Set the attributes
        self.reference = reference
        self.step = 0

        # Prepare x -> wavelength conversion and vice versa
        self.a0, self.a1 = calib
        self.b1 = 1 / self.a1
        self.b0 = -self.a0 / self.a1

        # Define x limits
        self.xlim = (scan.roi[0, 0], scan.roi[0, 1])
        self.wlim = (
            self.a1 * self.xlim[0] + self.a0,
            self.a1 * self.xlim[1] + self.a0,
        )

        # Current spectrum
        self.y = None
        self.yerr = None
        self.xe = None

        # Get the phex assignments
        self.phex = scan._phex.copy()

        # Assignment dialog input
        self.aqn = {"vpp": 0, "Jpp": 0}

        # Set up the main window
        super().__init__(scan, bins)

        # Add the fit action
        self.add_rect_selector(
            self.ax, self.assign, interactive=False, hint="Assign Peaks"
        )

        with _block_signals(*self.calc_options.values()):
            # Disable the calib action
            self.calc_options["calib"].setChecked(False)
            self.calc_options["calib"].setEnabled(False)

            # Activate and disable the montecarlo option
            self.calc_options["montecarlo"].setChecked(True)
            self.calc_options["montecarlo"].setEnabled(False)

            # Activate the other options
            self.calc_options["bkg"].setChecked(True)
            self.calc_options["qeff"].setChecked(True)

        # Plot the spectrum with the adjusted settings
        self.plot()

    def phex_index(self) -> str:
        return self.phex.iloc[self.step].index

    def phex_step(self) -> pd.Series:
        # Get the current phex assignment
        return self.phex.iloc[self.step].loc[["J", "Elp", "vp", "Jp"]]

    def find_phex(self) -> pd.DataFrame:
        idx = self.phex_index()

        if idx not in self.scan._phem.index:
            return pd.DataFrame()

        return self.scan._phem.loc[idx]

    def find_phem(self, phem_index: str) -> pd.Series:
        df = self.find_phex()

        if phem_index not in df.index:
            return None

        return df.loc[phem_index]

    def find_reference(self, phex: pd.Series) -> pd.DataFrame:
        ref = self.reference.copy()

        return pd.merge(ref, phex, how="inner")

    def plot(self, calc: bool = True) -> None:
        # Get the current phex assignment
        phex = self.phex_step()

        if calc:
            self.y, self.yerr, self.xe = self.scan.assigned_spectrum(
                phex["J"],
                phex["Elp"],
                phex["vp"],
                phex["Jp"],
                n_std=1,
                normalize=True,
                bins=self.bins,
                qeff=self.calc_options["qeff"].isChecked,
                bkg=self.calc_options["bkg"].isChecked,
                calib=False,
                uncertainties="montecarlo",
            )

        with ageplot.context(["age", "interactive"]):
            # Clear the axes
            self.ax.clear()

            # Set the title
            title = r"$X(v = 0, J = " + str(phex["J"]) + r") \rightarrow "
            title += phex["Elp"] + r"(v^\rpime = " + str(phex["vp"])
            title += r", J^\prime = " + str(phex["Jp"]) + ")$"
            self.ax.set_title(title)

            # Plot the spectrum
            self.ax.stairs(self.y, self.xe, color=ageplot.colors[1])

            # Plot the uncertainties
            self.ax.stairs(
                self.y + self.yerr,
                self.xe,
                baseline=self.y - self.yerr,
                color=ageplot.colors[1],
                fill=True,
                alpha=0.5,
            )

            # Set the limits
            self.ax.set_xlim(*self.xlim)
            self.ax.set_ylim(0, np.max(self.y) * 1.1)

            # Plot assignments
            phem = self.find_phex()

            # Plot the found assignments
            for row in phem.itertuples():
                # Define x values
                x = np.linspace(*self.xlim, 1000)

                # Calculate the y values and their uncertainties
                y, yerr = row["fit"](x)

                # Scale to the bin width
                dx = self.xe[1] - self.xe[0]
                y *= dx
                yerr *= dx

                # Plot the fit results
                self.ax.plot(x, y, color=ageplot.colors[0])
                self.ax.fill_between(
                    x, y - yerr, y + yerr, color=ageplot.colors[0], alpha=0.5
                )

            # Plot reference
            ref = self.find_reference(phex)

            # Select lines within the wavelength limits
            if not ref.empty:
                ref = ref.query("emi_energy >= @self.wlim[0]").query(
                    "emi_energy <= @self.wlim[1]"
                )

            # Plot the found reference
            for row in ref.itertuples():
                # Convert to detector position
                x = self.b1 * row["emi_energy"] + self.b0

                # Plot the reference
                self.ax.axvline(x, color="black", linestyle="--")

                # Create the text
                text = r"$v^{\prime\prime} = " + str(row["vpp"])
                text += r", J^{\prime\prime} = " + str(row["Jpp"]) + "$"

                self.ax.text(
                    x + 0.0004,
                    np.max(self.y),
                    text,
                    ha="left",
                    va="top",
                    rotation=90,
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
        phex_index = self.phex_index()

        # Prepare the data for the fit
        in_range = np.argwhere(
            (self.xe >= xr[0]) & (self.xe <= xr[1])
        ).flatten()

        xe = self.xe[in_range]
        y = self.y[in_range[:-1]]
        yerr = self.yerr[in_range[:-1]]

        n = np.stack((y, yerr**2), axis=-1)

        # Get the assignment for the first peak
        dialog = AssignmentDialog(self, self.aqn, title="Assign Peak 1")

        if dialog.exec():
            # Get the user input
            phem1_input = dialog.get_input()
            phem1_index = (
                str(phem1_input["vpp"]) + "," + str(phem1_input["Jpp"])
            )

            # Set the default signal model
            sig1_model = "Voigt"
            sig2_model = "None"

            # Check if the assignment exists
            phem1 = self.find_phem(phem1_index)

            if phem1 is None:
                phem1 = phem1_input

            else:
                phem1 = phem1.to_dict()
                sig1_model = phem1["fit"]

        else:
            return

        # Get the assignment for the second peak
        dialog = AssignmentDialog(self, self.aqn, title="Assign Peak 2")

        if dialog.exec():
            # Get the user input
            phem2_input = dialog.get_input()
            phem2_index = (
                str(phem2_input["vpp"]) + "," + str(phem2_input["Jpp"])
            )

            # Set the default signal model
            sig2_model = "Voigt"

            phem2 = self.find_phem(phem2_index)

            if phem2 is None:
                phem2 = phem2_input

            else:
                phem2 = phem2.to_dict()
                sig2_model = phem2["fit"]

        else:
            phem2 = None

        debug_fit = InteractiveFit(
            self,
            n,
            xe,
            sig=[sig1_model, sig2_model],
            bkg="Constant",
        )

        # Fit the data
        if debug_fit.exec():
            # Get the fit results
            res1, res2 = debug_fit.fit_result()

            # Don't save the assignment if the fit failed
            if res1 is None:
                return

            # Create the new row
            phem1["fit"] = res1

            # Save the assignment
            self.scan._phem.loc[phex_index].loc[phem1_index] = phem1

            if phem2 is not None and res2 is not None:
                # Create the new row
                phem2["fit"] = res2

                # Save the assignment
                self.scan._phem.loc[phex_index].loc[phem2_index] = phem2

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

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        n: NDArray,
        xe: NDArray,
        sig: tuple[str | FitModel, str | FitModel] = ["Voigt", "Voigt"],
        bkg: str = "Constant",
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

        # Set signal and background models
        if isinstance(sig[0], FitModel):
            self.sig1 = sig[0]
            sig[0] = self.sig1.name

        else:
            self.sig1 = None

        if isinstance(sig[1], FitModel):
            self.sig2 = sig[1]
            sig[1] = self.sig2.name

        else:
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
            QtWidgets.QSizePolicy.Policy.MinimumExpanding,
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
        self.prepare_fit(sig1=self.sig1, sig2=self.sig2)

    def update_fit_widget(self, widget: QtWidgets.QWidget) -> None:
        # Remove the old fit widget
        self.layout.removeWidget(widget)
        # Set the new fit widget
        self.fit_widget = widget
        self.layout.addWidget(self.fit_widget, 0, 0, 1, 4)

    def prepare_fit(
        self, sig1: FitModel | None = None, sig2: FitModel | None = None
    ) -> None:
        # Get the selected signal and background models
        if sig1 is None:
            sig1 = self.sig_comp1.currentText()
            sig1 = self.sig_models[sig1](self.xr)
            dloc1 = 0.4 * (self.xr[1] - self.xr[0]) + self.xr[0]

        else:
            sig1.xr = self.xr
            dloc1 = 0

        if sig2 is None:
            sig2 = self.sig_comp2.currentText()
            if sig2 != "None":
                sig2 = self.sig_models[sig2](self.xr)
                dloc2 = 0.6 * (self.xr[1] - self.xr[0]) + self.xr[0]

        else:
            sig2.xr = self.xr
            dloc2 = 0

        bkg = self.bkg_comp.currentText()

        # Remember current parameters and limits
        params_prev = self.params.copy()

        # Get the first signal model, starting values and limits
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
                return (
                    sig1.cdf(x, args[:idx1])
                    + sig2.cdf(x, args[idx1:idx2])
                    + bkg.cdf(x, args[idx2:])
                )

        # Shift the loc parameters if there are two signal components
        if self.sig2 is not None:
            self.params["loc1"] = dloc1
            self.params["loc2"] = dloc2

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

    def fit_result(self) -> tuple[FitModel | None, FitModel | None]:
        if not self.m.valid:
            return None, None

        # Get the covariance matrix
        cov = np.array(self.m.covariance)

        # Append 1 to the parameter names
        par1 = [f"{k}1" for k in self.sig1.par]

        # Get fitted parameter values and uncertainties
        self.sig1.val = np.array(self.m.values[par1])
        self.sig1.err = np.array(self.m.errors[par1])

        # Get the covariance matrix for sig1
        self.sig1.cov = cov[: len(par1), : len(par1)]

        if self.sig2 is None:
            return self.sig1, None

        # Append 2 to the parameter names
        par2 = [f"{k}2" for k in self.sig2.par]

        # Get fitted parameter values and uncertainties
        self.sig2.val = np.array(self.m.values[par2])
        self.sig2.err = np.array(self.m.errors[par2])

        # Get the covariance matrix for sig2
        idx1 = len(par1)
        idx2 = idx1 + len(par2)
        self.sig2.cov = cov[idx1:idx2, idx1:idx2]

        return self.sig1, self.sig2
