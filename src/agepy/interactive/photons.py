from __future__ import annotations
from typing import Sequence, Tuple, Dict
import warnings
from typing import TYPE_CHECKING
import numpy as np
from scipy.stats import norm as scipy_norm
import pandas as pd
from . util import import_qt_binding, import_jacobi_propagate
# Import PySide6 / PyQt6 modules
qt_binding, QtWidgets, QtCore, QtGui = import_qt_binding()
# Import propagate from jacobi
propagate = import_jacobi_propagate()
# Import internal modules
from agepy.interactive import AGEDataViewer
from agepy.interactive.ui import PhexDialog, PhemDialog, FitSetupDialog
from agepy.interactive.fitting import PhotonSpectrumFit, PhotonExcitationFit
from agepy import ageplot
# Import modules for type hinting
if TYPE_CHECKING:
    from agepy.spec.photons import Scan, QEffScan, EnergyScan
    from matplotlib.backend_bases import MouseEvent

__all__ = []

class AGEScanViewer(AGEDataViewer):
    """Show all spectra in a scan.

    """

    def __init__(self, scan: Scan, bins: int = 512) -> None:
        super().__init__()
        # Add plot to canvas
        self.add_plot()
        # Add the toolbar
        self.add_toolbar()
        # Add forward and backward buttons
        self.add_forward_backward_action(self.plot_previous, self.plot_next)
        # Get the data
        self.y = []
        self.err = []
        _, self.x = np.histogram([], bins=bins, range=(0, 1))
        for step in scan.steps:
            y, err = scan.spectrum_at(step, self.x)
            self.y.append(y)
            self.err.append(err)
        # Remember current step
        self.step = 0
        # Plot the first step
        self.plot(self.step)
    
    def plot(self, step: int) -> None:
        with ageplot.context(["age", "dataviewer"]):
            self.ax.clear()
            self.ax.stairs(self.y[step], self.x)
            self.canvas.draw()

    def plot_previous(self) -> None:
        self.step -= 1
        if self.step < 0:
            self.step = 0
        self.plot(self.step)

    def plot_next(self) -> None:
        self.step += 1
        if self.step >= len(self.y):
            self.step = len(self.y) - 1
        self.plot(self.step)


class QEffViewer(AGEScanViewer):
    """Fit peaks in a spectrum interactively.
    The fitting range can be set interactively by the user.

    """

    def __init__(self, scan: QEffScan) -> None:
        self.scan = scan
        super().__init__(self.scan, bins=1024)
        # Add the fit area button to select a region of interest and
        # then fit the data within that region
        self.add_rect_selector(self.ax, self.on_select, hint="Select Peak")
        # Add button to view results
        self.view_results = QtWidgets.QPushButton("View Results")
        self.view_results.setCheckable(True)
        self.view_results.clicked.connect(self.show_results)
        self.toolbar.addWidget(self.view_results)

    def on_select(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        # Fit in the selected region
        n, xe = self.scan._fit_in_range(self.step, (x1, x2))
        # Create the widget
        self.debug_fit = PhotonSpectrumFit(n, xe, sig=self.scan.model)
        self.debug_fit.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.debug_fit.destroyed.connect(lambda: self.on_fit_closed(self.step, self.debug_fit.m))
        self.debug_fit.show()

    def on_fit_closed(self, step, m):
        # Clear the selector
        self.selector.clear()
        self.scan._add_fit_result(step, m.values["s"], m.errors["s"], m.values["loc"])
        self.debug_fit = None
        self.plot_next()

    def show_results(self):
        if self.view_results.isChecked():
            with ageplot.context(["age", "dataviewer"]):
                self.ax.clear()
                self.scan.plot_eff(self.ax, color=ageplot.colors[0])
                self.canvas.draw()
        else:
            self.plot(self.step)


class PhexViewer(AGEDataViewer):

    def __init__(self,
        scan: EnergyScan,
        reference: pd.DataFrame,
        label: Dict,
        energy_range: float,
        simulation: pd.DataFrame,
    ) -> None:
        self.scan = scan
        self.reference = reference
        self.label = label
        self.simulation = simulation
        super().__init__()
        # Add emtpy figure
        self.add_plot()
        # Add the toolbar
        self.add_toolbar()
        # Add rectangle selector
        self.add_rect_selector(self.ax, self.assign, interactive=False)
        # Add Forward and Backward buttons
        self.add_forward_backward_action(self.prev, self.next)
        # Add Look-Up button
        self.add_lookup_action(self.look_up)
        # Prepare the data
        self.y, self.yerr, self.x = self.scan.counts(qeff=False)
        self.ymax = np.max(self.y)
        self.ymin = np.min(self.y)
        # Prepare assignments
        if self.scan.phex_assignments is None:
            self.scan.phex_assignments = pd.DataFrame(columns=["E", *self.label, "val", "err"])
        # Set the x limit
        self.dx = energy_range
        self.xlim = (self.x[0], self.x[0] + self.dx)
        # Plot the data
        with ageplot.context(["age", "dataviewer"]):
            self.plot()

    def plot(self) -> None:
        in_range = (self.x >= self.xlim[0]) & (self.x <= self.xlim[1])
        x = self.x[in_range]
        y = self.y[in_range]
        yerr = self.yerr[in_range]
        with ageplot.context(["age", "dataviewer"]):
            # Clear the plot
            self.ax.clear()
            # Plot the data
            self.ax.errorbar(x, y, yerr=yerr, xerr=0.0002, fmt="s",
                             color=ageplot.colors[0])
            # Plot the reference / simulation
            if self.simulation is not None:
                ref = self.simulation.query("E >= @self.xlim[0]").query(
                    "E <= @self.xlim[1]")
            else:
                ref = self.reference.query("E >= @self.xlim[0]").query(
                    "E <= @self.xlim[1]")
            for i, row in ref.iterrows():
                self.ax.axvline(row["E"], color="black", linestyle="--")
                text = ""
                for l in self.label:
                    if l in row:
                        text += f"{row[l]},"
                text = text[:-1]
                self.ax.text(row["E"] + 0.0002, self.ymax, text, ha="left",
                             va="top", rotation=90)
            # Plot the assignments
            fit = self.scan.phex_assignments.query(
                "E >= @self.xlim[0]").query("E <= @self.xlim[1]")
            x_fit = np.linspace(self.xlim[0], self.xlim[1], 1000)
            for i, row in fit.iterrows():
                val = row["val"]
                err = row["err"]
                y_fit, yerr_fit = propagate(lambda par: par[0] * scipy_norm.pdf(x_fit, *par[1:]), val, err**2)
                yerr_fit = np.sqrt(np.diag(yerr_fit))
                self.ax.plot(x_fit, y_fit, color=ageplot.colors[1])
                self.ax.fill_between(x_fit, y_fit - yerr_fit, y_fit + yerr_fit,
                                     color=ageplot.colors[1], alpha=0.5)
            self.ax.set_xlim(self.xlim)
            self.ax.set_ylim(self.ymin * 0.99, self.ymax * 1.05)
            self.ax.set_xlabel("Energy [eV]")
            self.ax.set_ylabel(r"Counts [arb.$\,$u.]")
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
        dialog = PhexDialog(self, self.label, title="Look-Up")
        if dialog.exec():
            exc = dialog.get_input()
            if self.simulation is not None:
                E = self.simulation.copy()
            else:
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
        y = self.y[in_range]
        yerr = self.yerr[in_range]

        # Get the assignments
        dialog = PhexDialog(self, self.label, title="Assign Peak 1")
        if dialog.exec():
            exc1 = dialog.get_input()
        else:
            return
        dialog = PhexDialog(self, self.label, title="Assign Peak 2")
        if dialog.exec():
            exc2 = dialog.get_input()
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
                constraint = (np.abs(E2["E"].iloc[0] - E1["E"].iloc[0]), 0.0005)
            self.debug_fit = PhotonExcitationFit(y, yerr, x, sig=["Gaussian", "Gaussian"], bkg="Constant", constrain_dE=constraint)
        else:
            exc2 = None
            self.debug_fit = PhotonExcitationFit(y, yerr, x, sig="Gaussian", bkg="Constant")

        # Fit the data
        self.debug_fit.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.debug_fit.destroyed.connect(lambda: self.on_fit_closed(self.debug_fit.m, exc1, exc2))
        self.debug_fit.show()

    def on_fit_closed(self, m, exc1, exc2):
        exc1["E"] = float(m.values["loc1"])
        exc1["val"] = np.array(m.values["s1", "loc1", "scale1"])
        exc1["err"] = np.array(m.errors["s1", "loc1", "scale1"])
        idx = self.scan.phex_assignments.index.max()
        if np.isnan(idx):
            idx = -1
        self.scan.phex_assignments.loc[idx + 1] = exc1
        if exc2 is not None:
            if "loc2" in m.parameters:
                exc2["E"] = float(m.values["loc2"])
                exc2["val"] = np.array(m.values["s2", "loc2", "scale2"])
                exc2["err"] = np.array(m.errors["s2", "loc2", "scale2"])
            else:
                cov = np.array(m.covariance)
                inds = [m.parameters.index(p) for p in ["loc1", "loc2_loc1"]]
                cov = cov[np.ix_(inds, inds)]
                loc2, loc2err = propagate(lambda par: par[0] + par[1], m.values["loc1", "loc2_loc1"], cov)
                exc2["E"] = loc2
                exc2["val"] = np.array([m.values["s2"], loc2, m.values["scale1"]])
                exc2["err"] = np.array([m.errors["s2"], loc2err, m.errors["scale1"]])
            self.scan.phex_assignments.loc[idx + 2] = exc2
        # Update the plot
        self.plot()
        self.debug_fit = None


class PhemViewer(AGEDataViewer):
    def __init__(self,
        scan: EnergyScan,
        reference: pd.DataFrame,
        qnum: Sequence[str],
        wl_range: Tuple[float, float],
        simulation: pd.DataFrame,
    ) -> None:
        super().__init__()
        self.scan = scan
        self.reference = reference
        self.qnum = qnum
        self.wl_range = wl_range
        self.simulation = simulation
        # Add emtpy figure
        self.add_plot()
        # Add the toolbar
        self.add_toolbar()
        # Add rectangle selector
        self.add_rect_selector(self.ax, self.assign, interactive=False, hint="Assign Peak")
        # Add Forward and Backward buttons
        self.add_forward_backward_action(self.prev, self.next)
        # Add Look-Up button
        self.add_lookup_action(self.look_up)
        # Add button to view results
        self.view_results = QtWidgets.QPushButton("View Results")
        self.view_results.setCheckable(True)
        self.view_results.clicked.connect(self.show_results)
        self.toolbar.addWidget(self.view_results)
        # Current index
        self.phex = list(self.scan.phex_assignments.keys())
        self.idx = 0
        # Define bins
        self.edges = np.histogram([], bins=512, range=(0, 1))[1]
        # Plot the first spectrum
        self.plot()

    def plot(self):
        with ageplot.context(["age", "dataviewer"]):
            self.ax.clear()
            # Plot data
            spec, err = self.scan.assigned_spectrum(
                self.phex[self.idx], self.edges, calib=False)
            self.ax.stairs(spec, self.edges, color=ageplot.colors[1])
            xlim = (self.scan.roi["x"]["min"], self.scan.roi["x"]["max"])
            self.ax.set_xlim(*xlim)
            self.ax.set_ylim(0, np.max(spec) * 1.1)
            self.ax.set_title(f"{self.phex[self.idx]}")
            # Plot reference
            exc = self.phex[self.idx]
            ref = self.reference.query("E >= 130").query(
                "E <= 170").query("El == @exc[0]").query(
                    "vp == @exc[1]").query("Jp == @exc[2]")
            for i, row in ref.iterrows():
                x = row["E"] * self.wl_range[1] + self.wl_range[0]
                self.ax.axvline(x, color="black", linestyle="--")
                text = f"{row["vpp"]},{row["Jpp"]}"
                self.ax.text(x + 0.0002, np.max(spec), text, ha="left",
                             va="top", rotation=90)
            # Plot simulated spectrum
            if self.simulation is not None:
                x = np.linspace(135, 170, 1000)
                y = self.simulation.spectrum(self.phex[self.idx], x, 0.01, "nm")
                y = y / np.max(y) * np.max(spec)
                # Scale x to xlim
                x = x * self.wl_range[1] + self.wl_range[0]
                self.ax.plot(x, y, color=ageplot.colors[0])
            self.canvas.draw_idle()

    def next(self):
        if self.idx + 1 >= len(self.phex):
            return
        self.idx += 1
        self.plot()

    def prev(self):
        if self.idx - 1 < 0:
            return
        self.idx -= 1
        self.plot()

    def look_up(self):
        dialog = PhexDialog(self)
        if dialog.exec():
            exc = dialog.get_input()
            if tuple(exc) in self.phex:
                self.idx = self.phex.index(tuple(exc))
                self.plot()

    def show_results(self):
        if self.view_results.isChecked():
            with ageplot.context(["age", "dataviewer"]):
                self.ax.clear()
                fit_wl = []
                calib_wl = []
                for exc in self.scan.phem_assignments:
                    for rlx, res in self.scan.phem_assignments[tuple(exc)].items():
                        line = self.reference.query("El == @exc[0]").query(
                            "vp == @exc[1]").query("Jp == @exc[2]").query(
                            "vpp == @rlx[0]").query("Jpp == @rlx[1]")
                        if line.empty:
                            continue
                        val, err = res
                        fit_wl.append([val, err])
                        calib_wl.append(line["E"].iloc[0])
                fit_wl = np.array(fit_wl)
                calib_wl = np.array(calib_wl)
                # Sort by calibration wavelength
                idx = np.argsort(calib_wl)
                calib_wl = calib_wl[idx]
                fit_wl = fit_wl[idx]
                # Import the fitting modules
                try:
                    from iminuit import Minuit
                    from iminuit.cost import LeastSquares

                except ImportError:
                    raise ImportError("iminuit and numba-stats is required for fitting.")

                # Define the Model
                def model(x, a0, a1):
                    return a1 * x + a0
                # Create the cost function
                cost = LeastSquares(
                    calib_wl, fit_wl[:,0], fit_wl[:,1], model)
                # Initialize the minimizer
                a1_start = (fit_wl[1,0] - fit_wl[0,0]) / (calib_wl[1] - calib_wl[0])
                a0_start = fit_wl[0,0] - a1_start * calib_wl[0]
                m = Minuit(cost, a0=a0_start, a1=a1_start)
                m.migrad()
                print(m)
                self.ax.errorbar(calib_wl, fit_wl[:,0], yerr=fit_wl[:,1],
                    fmt="s", color=ageplot.colors[0], markersize=1.5,
                    label="Assign. Phex")
                self.ax.plot(calib_wl, cost.prediction(m.values),
                    color=ageplot.colors[1], label="Lin. Regr.")
                chi2ndof = m.fmin.reduced_chi2
                self.ax.legend(title=r"$\chi^2\;/\;$ndof = " + f"{chi2ndof:.2f}")
                self.canvas.draw()
                a0, a1 = m.values["a0"], m.values["a1"]
                b1 = 1 / a1
                b0 = -a0 / a1
                self.scan.calib = (b0, b1)
        else:
            self.plot()

    def assign(self, eclick: MouseEvent, erelease: MouseEvent):
        xr = (eclick.xdata, erelease.xdata)
        # Get the data in the selected range
        spec, err = self.scan.assigned_spectrum(
            self.phex[self.idx], self.edges, calib=False)
        in_range = np.argwhere((self.edges >= xr[0]) & (self.edges <= xr[1])).flatten()
        x = self.edges[in_range]
        y = spec[in_range[:-1]]
        yerr = err[in_range[:-1]]

        # Fit the data
        n = np.stack((y, yerr**2), axis=-1)
        self.debug_fit = PhotonSpectrumFit(n, x, sig=["Gaussian", "Gaussian"], bkg="Constant")
        self.debug_fit.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.debug_fit.destroyed.connect(lambda: self.on_fit_closed(self.debug_fit.m))
        self.debug_fit.show()

    def on_fit_closed(self, m):
        self.debug_fit = None
        dialog = PhemDialog(self)
        if dialog.exec():
            loc = "loc" if "loc" in m.parameters else "loc1"
            phem = tuple(dialog.get_input())
            phex = tuple(self.phex[self.idx])
            if phex in self.scan.phem_assignments:
                self.scan.phem_assignments[phex][phem] = (m.values[loc], m.errors[loc])
            else:
                self.scan.phem_assignments[phex] = {phem: (m.values[loc], m.errors[loc])}
        dialog = PhemDialog(self)
        if dialog.exec() and "loc2" in m.parameters:
            phem = tuple(dialog.get_input())
            phex = tuple(self.phex[self.idx])
            if phex in self.scan.phem_assignments:
                self.scan.phem_assignments[phex][phem] = (m.values["loc2"], m.errors["loc2"])
            else:
                self.scan.phem_assignments[phex] = {phem: (m.values["loc2"], m.errors["loc2"])}
