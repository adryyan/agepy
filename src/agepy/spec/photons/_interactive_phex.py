from __future__ import annotations

try:
    from PySide6 import QtWidgets, QtCore, QtGui

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
import pandas as pd
import matplotlib.pyplot as plt

from agepy.interactive import MainWindow
from ._assignment_dialog import AssignmentDialog
from ._interactive_fit import Gaussian, Constant
from agepy import ageplot

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.backend_bases import MouseEvent
    from numpy.typing import NDArray, ArrayLike
    from .energy_scan import EnergyScan
    from ._interactive_fit import FitModel


class AssignPhex(MainWindow):
    def __init__(
        self,
        scan: EnergyScan,
        reference: pd.DataFrame,
        energy_range: float,
    ) -> None:
        # Set the scan and reference data
        self.scan = scan
        self.reference = reference

        # Initialize the parent class
        super().__init__()

        # Set up the main window
        self.add_plot()
        self.add_toolbar()
        self.add_lookup_action(self.look_up)
        self.add_forward_backward_action(self.prev, self.next)
        self.add_rect_selector(
            self.ax, self.assign, interactive=False, hint="Assign Excitation"
        )

        self.aqn = {"J": 0, "Elp": ["B"], "vp": 0, "Jp": 0}

        # Prepare the data
        self.y, self.yerr, self.x = self.scan.counts(bkg=False)
        self.xerr = self.scan.energy_uncertainty
        self.ymax = np.max(self.y)
        self.ymin = np.min(self.y)

        # Set the x limit
        self.dx = energy_range
        self.xlim = (self.x[0], self.x[0] + self.dx)

        # Plot the data
        self.plot()

    def plot(self) -> None:
        # Get the data in the selected range
        in_range = (self.x >= self.xlim[0]) & (self.x <= self.xlim[1])
        x = self.x[in_range]
        xerr = self.xerr[in_range]
        y = self.y[in_range]
        yerr = self.yerr[in_range]

        # Plot the data and reference
        with ageplot.context(["age", "interactive"]):
            # Clear the plot
            self.ax.clear()

            # Plot the data
            self.ax.errorbar(
                x, y, yerr=yerr, xerr=xerr, fmt="s", color=ageplot.colors[0]
            )

            # Plot the reference
            ref = self.reference.query("E >= @self.xlim[0]").query(
                "E <= @self.xlim[1]"
            )

            for row in ref.itertuples():
                # Plot the reference energy
                self.ax.axvline(
                    row["exc_energy"], color="black", linestyle="--"
                )

                # Create the text
                text = r"$X(v = 0, J = " + str(row["J"]) + r") \rightarrow "
                text += row["Elp"] + r"(v^\rpime = " + str(row["vp"])
                text += r", J^\prime = " + str(row["Jp"]) + ")$"

                # Plot the label
                self.ax.text(
                    row["exc_energy"] + self.dx * 0.001,
                    self.ymax,
                    text,
                    ha="left",
                    va="top",
                    rotation=90,
                )

            # Plot the assignments
            fit = self.scan._phex.query("exc_energy >= @self.xlim[0]").query(
                "exc_energy <= @self.xlim[1]"
            )

            # Define x values
            x_fit = np.linspace(self.xlim[0], self.xlim[1], 1000)

            for row in fit.itertuples():
                # Evaluate at the x values and propagate the uncertainties
                y_fit, yerr_fit = row["fit"](x_fit)

                # Plot the fit
                self.ax.plot(x_fit, y_fit, color=ageplot.colors[1])
                self.ax.fill_between(
                    x_fit,
                    y_fit - yerr_fit,
                    y_fit + yerr_fit,
                    color=ageplot.colors[1],
                    alpha=0.5,
                )

            # Set the plot limits and labels
            self.ax.set_xlim(self.xlim)
            self.ax.set_ylim(self.ymin * 0.99, self.ymax * 1.05)
            self.ax.set_xlabel("Energy [eV]")
            self.ax.set_ylabel(r"Counts [arb.$\,$u.]")

            # Update the canvas
            self.canvas.draw_idle()

    def next(self):
        E_min = self.xlim[1] - 0.2 * self.dx
        E_max = E_min + self.dx

        if E_min >= self.x[-1]:
            return

        self.xlim = (E_min, E_max)
        self.plot()

    def prev(self):
        E_max = self.xlim[0] + 0.2 * self.dx
        E_min = E_max - self.dx

        if E_max <= self.x[0]:
            return

        self.xlim = (E_min, E_max)
        self.plot()

    def look_up(self):
        dialog = AssignmentDialog(self, self.aqn, title="Look-Up")

        if dialog.exec():
            exc = dialog.get_input()
            E = self.reference.copy()

            for l, val in exc.items():  # noqa B007
                query_str = f"{l} == @val"
                E.query(query_str, inplace=True)

            if E.empty:
                return

            else:
                E = E["exc_energy"].iloc[0]

            E_min, E_max = E - self.dx * 0.5, E + self.dx * 0.5

            if E_min >= self.x[-1] or E_max <= self.x[0]:
                return

            self.xlim = (E_min, E_max)
            self.plot()

    def assign(self, eclick: MouseEvent, erelease: MouseEvent):
        xr = (eclick.xdata, erelease.xdata)

        # Get the data in the selected range
        in_range = (self.x >= xr[0]) & (self.x <= xr[1])
        x = self.x[in_range]
        xerr = self.xerr[in_range]
        y = self.y[in_range]
        yerr = self.yerr[in_range]

        # Define starting values for the fit
        val_start1 = {}
        val_start2 = {}

        # Get the assignments
        dialog = AssignmentDialog(self, self.aqn, title="Assign Peak I")
        if dialog.exec():
            phex1 = dialog.get_input()
            phex1_index = str(phex1["J"]) + "," + phex1["Elp"] + ","
            phex1_index += str(phex1["vp"]) + "," + str(phex1["Jp"])

            if phex1_index in self.scan._phex.index:
                # Get the starting values
                val_start1 = self.scan._phex.loc[phex1_index]["fit"]
                val_start1 = dict(zip(["s1", "loc1", "scale1"], val_start1))

        else:
            return

        dialog = AssignmentDialog(self, self.aqn, title="Assign Peak II")

        if dialog.exec():
            phex2 = dialog.get_input()
            phex2_index = str(phex2["J"]) + "," + phex2["Elp"] + ","
            phex2_index += str(phex2["vp"]) + "," + str(phex2["Jp"])

            if phex2_index in self.scan._phex.index:
                # Get the starting values
                val_start2 = self.scan._phex.loc[phex1_index]["fit"]
                val_start2[1] -= val_start1["loc1"]
                val_start2 = {
                    "s2": val_start2[0],
                    "loc2_loc1": val_start2[1],
                }

            # Prepare constraint of the energy difference
            E1 = self.reference.copy()
            for q, val in phex1.items():  # noqa B007
                query_str = f"{q} == @val"
                E1.query(query_str, inplace=True)

            E2 = self.reference.copy()
            for q, val in phex2.items():  # noqa B007
                query_str = f"{q} == @val"
                E2.query(query_str, inplace=True)

            if E1.empty or E2.empty:
                constraint = None

            else:
                diff = np.abs(
                    E2["exc_energy"].iloc[0] - E1["exc_energy"].iloc[0]
                )
                constraint = (diff, np.mean(xerr) * 2)

            debug_fit = InteractiveFit(
                self,
                y,
                yerr,
                x,
                xerr,
                sig=["Gaussian", "Gaussian"],
                bkg="Constant",
                constrain_dE=constraint,
                **val_start1,
                **val_start2,
            )

        else:
            phex2 = None
            debug_fit = InteractiveFit(
                self,
                y,
                yerr,
                x,
                xerr,
                sig="Gaussian",
                bkg="Constant",
                **val_start1,
            )

        # Fit the data
        if debug_fit.exec():
            # Get the fit results
            res1, res2 = debug_fit.fit_result()

            # Don't save the assignment if the fit failed
            if res1 is None:
                return

            # Create the new row
            phex1["fit"] = res1
            phex1["exc_energy"] = res1.val[1]

            # Save the assignment
            self.scan._phex.loc[phex1_index] = phex1

            if phex2 is not None and res2 is not None:
                # Create the new row
                phex2["fit"] = res2
                phex2["exc_energy"] = res2.val[1]

                # Save the assignment
                self.scan._phex.loc[phex2_index] = phex2

            # Close pyplot figures
            plt.close("all")

            # Update the plot
            self.plot()


class InteractiveFit(QtWidgets.QDialog):
    sig_models = {
        "Gaussian": lambda xr: Gaussian(xr),
    }

    bkg_models = {
        "None": None,
        "Constant": lambda xr: Constant(xr),
    }

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        y: NDArray,
        yerr: NDArray,
        x: NDArray,
        xerr: NDArray,
        sig1: str | FitModel = "Gaussian",
        sig2: str | FitModel = "Gaussian",
        bkg: str = "Constant",
        constraint: tuple[float, float] | None = None,
    ) -> None:
        # Initialize fit data
        self.y = y
        self.yerr = yerr
        self.x = x
        self.xerr = xerr

        # Set the x range
        self.xr = (x[0], x[-1])

        # Define starting values and limits
        self.s_start = np.max(y) * (x[1] - x[0])
        self.s_limit = np.max(y) * (x[1] - x[0]) * 5

        # Set the constraint
        self.constraint = constraint

        # Set signal and background models
        if isinstance(sig1, FitModel):
            self.sig1 = sig1
            sig1 = self.sig1.name

        else:
            self.sig1 = None

        if isinstance(sig2, FitModel):
            self.sig2 = sig2
            sig2 = self.sig2.name

        else:
            self.sig2 = None

        self.bkg = None

        # Initialize the parameters
        self.params = {}

        # Initialize the parent class
        super().__init__(parent)
        self.setWindowTitle("Photon Excitation Fit")
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
            list(self.sig_models.keys()).index(sig1)
        )
        self.sig_comp1.currentIndexChanged.connect(self.prepare_fit)
        self.signal_layout.addWidget(self.sig_comp1)

        # Create ComboBox for the second signal component
        self.sig_comp2 = QtWidgets.QComboBox()
        self.sig_comp2.addItems(list(self.sig_models.keys()) + ["None"])
        self.sig_comp2.setCurrentIndex(
            (list(self.sig_models.keys()) + ["None"]).index(sig2)
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
            self.sig1 = self.sig_models[sig1](self.xr)
            dloc1 = 0.1 * (self.xr[1] - self.xr[0])

        else:
            sig1.xr = self.xr
            dloc1 = 0

        if sig2 is None:
            sig2 = self.sig_comp2.currentText()

            if sig2 == "None":
                self.sig2 = None

            else:
                self.sig2 = self.sig_models[sig2](self.xr)
                dloc2 = 0.1 * (self.xr[1] - self.xr[0])

        else:
            sig2.xr = self.xr
            dloc2 = 0

        bkg = self.bkg_comp.currentText()

        if bkg != "None":
            bkg = self.bkg_models[bkg](self.xr)

        # Remember current parameters and limits
        params_prev = self.params.copy()

        # Get the first signal model, starting values and limits
        par1 = self.sig1.start_val(self.s_start)
        lim1 = self.sig1.limits(self.s_limit)

        # Append a 1 to the parameter names
        self.params = {f"{k}1": v for k, v in par1.items()}
        limits = {f"{k}1": v for k, v in lim1.items()}

        n1 = len(par1)
        n2 = 0
        n3 = 0

        if sig2 != "None":
            # Get the model and parameters of the second signal component
            par2 = self.sig2.start_val(self.s_start)
            lim2 = self.sig2.limits(self.s_limit)

            # Append a 2 to the parameter names
            par2 = {f"{k}2": v for k, v in par2.items()}
            lim2 = {f"{k}2": v for k, v in lim2.items()}

            # Shift loc1
            self.params["loc1"] -= dloc1

            # Convert loc2 to (loc2 - loc1)
            par2["loc2"] = par2["loc2"] - self.params["loc1"] + dloc2
            lim2["loc2"] = (0, self.xr[1] - self.xr[0])

            # Combine the parameters and limits
            self.params = {**self.params, **par2}
            limits = {**limits, **lim2}

            n2 = len(par2)

        if bkg != "None":
            # Get the background model and parameters
            par3 = bkg.start_val()
            lim3 = bkg.limits()

            # Combine the parameters and limits
            self.params = {**self.params, **par3}
            limits = {**limits, **lim3}

            n3 = len(par3)

        def model(x, args):
            y = self.sig1.pdf(x, args[:n1])

            if n2 > 0:
                y += self.sig2.pdf(x, args[n1 : n2 + n1])

            if n3 > 0:
                y += self.bkg.pdf(x, args[n1 + n2 :])

            return y

        def derivative(x, args):
            dy = self.sig1.der(x, args[:n1])

            if n2 > 0:
                dy += self.sig2.der(x, args[n1 : n1 + n2])

            return dy

        # Keep previous parameters and limits if possible
        for par in params_prev:
            if par in self.params:
                self.params[par] = self.m.values[par]

        def _cost(par):
            y_var = self.yerr**2 + (derivative(self.x, par) * self.xerr) ** 2
            result = np.sum((self.y - model(self.x, par)) ** 2 / y_var)

            return result

        class LeastSquaresXY(cost.LeastSquares):
            def _value(self, args: ArrayLike) -> float:
                return _cost(args)

        # Update the cost function
        c = LeastSquaresXY(self.x, self.y, self.yerr, model)

        if self.constraint is not None and sig2 != "None":
            c += cost.NormalConstraint("loc2", *self.constraint)

        # Update the Minuit object
        self.m = Minuit(
            c, *list(self.params.values()), name=list(self.params.keys())
        )

        # Set the limits
        for par, lim in limits.items():
            self.m.limits[par] = lim

        def plot(args):
            plt.errorbar(
                self.x, self.y, yerr=self.yerr, xerr=self.xerr, fmt="ok"
            )
            x = np.linspace(self.xr[0], self.xr[1], 1000)
            plt.plot(x, model(x, args))

        # Update the fit widget
        fit_widget = make_widget(self.m, plot, {}, False, False)

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
        self.sig1.cov = cov[: len(par1), : len(par1)].copy()

        if self.sig2 is None:
            return self.sig1, None

        # Append 2 to the parameter names
        par2 = [f"{k}2" for k in self.sig2.par]

        # Get fitted parameter values and uncertainties
        self.sig2.val = np.array(self.m.values[par2])
        self.sig2.err = np.array(self.m.errors[par2])

        # Add loc1 to loc2
        idx1 = par1.index("loc1")
        idx2 = par2.index("loc2")
        val = [self.sig1.val[idx1], self.sig2.val[idx2]]
        cov = cov[np.ix_([idx1, idx2], [idx1, idx2])]

        loc2, loc2err = propagate(lambda x: x[0] + x[1], val, cov)

        self.sig2.val[idx2] = loc2
        self.sig2.err[idx2] = np.sqrt(loc2err)
        self.sig2.cov = self.sig2.err**2

        return self.sig1, self.sig2
