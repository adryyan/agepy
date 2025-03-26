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
    from numba_stats import norm

except ImportError:
    raise ImportError("iminuit and jax required for fitting.")

import numpy as np
import numba as nb
from jacobi import propagate
import pandas as pd
import matplotlib.pyplot as plt

# Import internal modules
from agepy.spec.interactive.assignment_dialog import AssignmentDialog
from agepy.interactive import MainWindow
from agepy import ageplot

# Import modules for type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence, Tuple, Dict, Union
    from matplotlib.backend_bases import MouseEvent
    from numpy.typing import NDArray
    from agepy.spec.photons import EnergyScan

__all__ = ["AssignPhex"]


class AssignPhex(MainWindow):

    def __init__(self,
        scan: EnergyScan,
        reference: pd.DataFrame,
        label: Dict[str, Union[Sequence[str], int]],
        energy_range: float,
    ) -> None:
        # Set the scan and reference data
        self.scan = scan
        self.reference = reference
        self.label = label

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

        # Prepare the data
        self.y, self.yerr, self.x = self.scan.counts(qeff=False)
        self.xerr = self.scan.energy_uncertainty
        self.ymax = np.max(self.y)
        self.ymin = np.min(self.y)

        # Prepare assignments
        if self.scan._phex_assignments is None:
            self.scan._phex_assignments = pd.DataFrame(
                columns=["E", *self.label, "val", "err"]
            )

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
            ref = self.reference.query(
                "E >= @self.xlim[0]").query("E <= @self.xlim[1]")

            for i, row in ref.iterrows():
                # Plot the reference energy
                self.ax.axvline(row["E"], color="black", linestyle="--")

                # Create the text
                text = ""
                for label in self.label:
                    if label in row:
                        text += f"{row[label]},"

                # Remove the last comma
                text = text[:-1]

                # Plot the label
                self.ax.text(
                    row["E"] + self.dx * 0.001, self.ymax, text, ha="left",
                    va="top", rotation=90
                )

            # Plot the assignments
            fit = self.scan._phex_assignments.query(
                "E >= @self.xlim[0]").query("E <= @self.xlim[1]")
            
            # Define x values
            x_fit = np.linspace(self.xlim[0], self.xlim[1], 1000)

            for i, row in fit.iterrows():
                # Get the fit results
                val = row["val"]
                err = row["err"]

                # Evaluate at the x values and propagate the uncertainties
                y_fit, yerr_fit = propagate(
                    lambda par: par[0] * norm.pdf(x_fit, *par[1:]), val, err**2
                )
                yerr_fit = np.sqrt(np.diag(yerr_fit))

                # Plot the fit
                self.ax.plot(x_fit, y_fit, color=ageplot.colors[1])
                self.ax.fill_between(
                    x_fit, y_fit - yerr_fit, y_fit + yerr_fit,
                    color=ageplot.colors[1], alpha=0.5
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
        dialog = AssignmentDialog(self, self.label, title="Look-Up")
        if dialog.exec():
            exc = dialog.get_input()
            E = self.reference.copy()

            for l, val in exc.items():
                query_str = f"{l} == @val"
                E.query(query_str, inplace=True)

            if E.empty:
                return

            else:
                E = E["E"].iloc[0]

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
        val_start1 = None
        val_start2 = None

        # Get the assignments
        dialog = AssignmentDialog(self, self.label, title="Assign Peak I")
        if dialog.exec():
            exc1 = dialog.get_input()

            # Find the index where to save the assignment
            df = self.scan._phex_assignments.copy()
            for l, val in exc1.items():
                if df.empty:
                    break

                df.query(f"{l} == @val", inplace=True)

            if df.empty:
                idx1 = self.scan._phex_assignments.index.max()
                if np.isnan(idx1):
                    idx1 = 0

                else:
                    idx1 += 1

            else:
                idx1 = df.index[0]

                # Get the starting values
                val_start1 = df["val"].iloc[0]
                val_start1 = dict(zip(["s1", "loc1", "scale1"], val_start1))

        else:
            return

        dialog = AssignmentDialog(self, self.label, title="Assign Peak II")
        if dialog.exec():
            exc2 = dialog.get_input()

            # Find the index where to save the assignment
            df = self.scan._phex_assignments.copy()
            for l, val in exc2.items():
                if df.empty:
                    break

                df = df.query(f"{l} == @val")

            if df.empty:
                idx2 = idx1 + 1

            else:
                idx2 = df.index[0]

                # Get the starting values
                val_start2 = df["val"].iloc[0]
                val_start2[1] -= val_start1["loc1"]
                val_start2 = {
                    "s2": val_start2[0], "loc2_loc1": val_start2[1],
                }

            # Prepare constraint of the energy difference
            E1 = self.reference.copy()
            for l, val in exc1.items():
                query_str = f"{l} == @val"
                E1.query(query_str, inplace=True)

            E2 = self.reference.copy()
            for l, val in exc2.items():
                query_str = f"{l} == @val"
                E2.query(query_str, inplace=True)

            if E1.empty or E2.empty:
                constraint = None

            else:
                diff = np.abs(E2["E"].iloc[0] - E1["E"].iloc[0])
                constraint = (diff, np.mean(xerr) * 2)

            debug_fit = InteractiveFit(
                self, y, yerr, x, xerr, sig=["Gaussian", "Gaussian"],
                bkg="Constant", constrain_dE=constraint, **val_start1,
                **val_start2
            )

        else:
            exc2 = None
            debug_fit = InteractiveFit(
                self, y, yerr, x, xerr, sig="Gaussian", bkg="Constant",
                **val_start1
            )

        # Fit the data
        if debug_fit.exec():
            m = debug_fit.m

            # Create the new row
            exc1["E"] = float(m.values["loc1"])
            exc1["val"] = np.array(m.values["s1", "loc1", "scale1"])
            exc1["err"] = np.array(m.errors["s1", "loc1", "scale1"])

            # Save the assignment
            self.scan._phex_assignments.loc[idx1] = exc1

            # Process the second assignment
            if exc2 is not None:
                # Create the new row
                cov = np.array(m.covariance)
                inds = [m.parameters.index(p) for p in ["loc1", "loc2_loc1"]]
                cov = cov[np.ix_(inds, inds)]
                loc2, loc2err = propagate(
                    lambda par: par[0] + par[1],
                    m.values["loc1", "loc2_loc1"], cov
                )

                exc2["E"] = loc2
                exc2["val"] = np.array(
                    [m.values["s2"], loc2, m.values["scale1"]]
                )
                exc2["err"] = np.array(
                    [m.errors["s2"], loc2err, m.errors["scale1"]]
                )

                # Save the assignment
                self.scan._phex_assignments.loc[idx2] = exc2

            # Close the fit plot
            plt.close()

            # Update the plot
            self.plot()


class InteractiveFit(QtWidgets.QDialog):

    def __init__(self,
        parent: QtWidgets.QWidget,
        y: NDArray,
        yerr: NDArray,
        x: NDArray,
        xerr: NDArray,
        sig: Union[str, Tuple[str, str]] = "Gaussian",
        bkg: str = "Constant",
        constrain_dE: Tuple[float, float] = None,
        **start_values
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
        self.start_values = start_values

        # Set the constraint
        self.constrain_dE = constrain_dE

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
            QtWidgets.QSizePolicy.Policy.MinimumExpanding
        )

        # Create signal model selection widget
        self.signal_group = QtWidgets.QGroupBox("Signal Model")
        self.signal_group.setSizePolicy(size_policy)
        self.signal_layout = QtWidgets.QHBoxLayout(self.signal_group)

        # Create ComboBox for the first signal component
        self.sig_comp1 = QtWidgets.QComboBox()
        self.sig_comp1.setSizePolicy(size_policy)
        self.sig_comp1.addItems(self.sig_models.keys())
        self.sig_comp1.setCurrentIndex(list(self.sig_models.keys()).index(sig[0]))
        self.sig_comp1.currentIndexChanged.connect(self.prepare_fit)
        self.signal_layout.addWidget(self.sig_comp1)

        # Create ComboBox for the second signal component
        self.sig_comp2 = QtWidgets.QComboBox()
        self.sig_comp2.setSizePolicy(size_policy)
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
        self.bkg_comp.setSizePolicy(size_policy)
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
        # Remember current parameters and limits
        params_prev = self.params.copy()

        #
        nconstraint = None

        # Get the model and derivative
        if self.sig_comp2.currentText() == "None":
            model, derivative, self.params, limits = self.gaussian()

        else:
            model, derivative, self.params, limits = self.constrained_gaussians()

            # Define the constraint if available
            if self.constrain_dE is not None:
                nconstraint = cost.NormalConstraint(
                    "loc2_loc1", *self.constrain_dE
                )

        # Overwrite with given starting values
        for par, val in self.start_values.items():
            if par in self.params:
                self.params[par] = val

        # Keep previous parameters values
        for par in params_prev:
            if par in self.params:
                self.params[par] = self.m.values[par]


        # Define the cost function
        x = self.x
        xerr = self.xerr
        y = self.y
        yerr = self.yerr

        @nb.njit
        def _cost(par):
            y_var = yerr**2 + (derivative(x, *par) * xerr) ** 2
            result = np.sum((y - model(x, *par)) ** 2 / y_var)
            return result

        class LeastSquaresXY(cost.LeastSquares):

            def _value(self, args: Sequence[float]) -> float:
                return _cost(args)

        # Update the cost function
        if nconstraint is None:
            c = LeastSquaresXY(self.x, self.y, self.yerr, model)
        else:
            c = LeastSquaresXY(self.x, self.y, self.yerr, model) + nconstraint

        # Update the Minuit object
        self.m = Minuit(c, *list(self.params.values()), name=list(self.params.keys()))

        # Set the limits
        for par, lim in limits.items():
            self.m.limits[par] = lim

        def plot(args):
            plt.errorbar(self.x, self.y, yerr=self.yerr, xerr=self.xerr, fmt="ok")
            x = np.linspace(self.xr[0], self.xr[1], 1000)
            plt.plot(x, model(x, *args))

        # Update the fit widget
        fit_widget = make_widget(self.m, plot, {}, False, False)

        # Perform the fit
        fit_widget.fit_button.click()

        # Update the layout
        self.update_fit_widget(fit_widget)

    def _init_model_list(self) -> None:
        self.sig_models = {
            "Gaussian": self.gaussian,
        }
        self.bkg_models = {
            "Constant": None,
        }

    def gaussian(self) -> Tuple[callable, callable, dict, dict]:
        dx = self.xr[1] - self.xr[0]
        params = {
            "s1": self.s_start, "loc1": self.xr[0] + 0.5 * dx,
            "scale1": 0.1 * dx, "b": self.x[1] - self.x[0]
        }
        limits = {
            "s1": (0, self.s_limit), "loc1": self.xr,
            "scale1": (0.0001 * dx, 0.5 * dx), "b": (0, None)
        }

        @nb.njit
        def model(x, s1, loc1, scale1, b):
            return s1 * norm.pdf(x, loc1, scale1) + b

        @nb.njit
        def derivative(x, s1, loc1, scale1, b):
            return -s1 * norm.pdf(x, loc1, scale1) * (x - loc1) / scale1**2

        return model, derivative, params, limits

    def constrained_gaussians(self) -> Tuple[callable, callable, dict, dict]:
        dx = self.xr[1] - self.xr[0]
        params = {
            "s1": self.s_start, "s2": self.s_start,
            "loc1": self.xr[0] + 0.35 * dx,
            "loc2_loc1": 0.35 * dx,
            "scale1": 0.1 * dx, "b": self.x[1] - self.x[0]
        }
        limits = {
            "s1": (0, self.s_limit), "s2": (0, self.s_limit),
            "loc1": self.xr, "loc2_loc1": (0, dx),
            "scale1": (0.0001 * dx, 0.5 * dx), "b": (0, None)
        }

        @nb.njit
        def model(x, s1, s2, loc1, loc2_loc1, scale1, b):
            return (s1 * norm.pdf(x, loc1, scale1)
                    + s2 * norm.pdf(x, loc1 + loc2_loc1, scale1) + b)

        @nb.njit
        def derivative(x, s1, s2, loc1, loc2_loc1, scale1, b):
            return (-s1 * norm.pdf(x, loc1, scale1) * (x - loc1) / scale1**2
                    - s2 * norm.pdf(x, loc1 + loc2_loc1, scale1) * (x - loc1 - loc2_loc1) / scale1**2)

        return model, derivative, params, limits
