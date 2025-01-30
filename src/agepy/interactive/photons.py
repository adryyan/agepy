from __future__ import annotations
from typing import Sequence, Tuple
import warnings
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
# Import PySide6 / PyQt6 modules
from . util import import_qt_binding
qt_binding, QtWidgets, QtCore, QtGui = import_qt_binding()
# Import internal modules
from agepy.interactive import AGEDataViewer
from agepy.interactive.ui import PhexDialog
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
        m = self.scan._fit_in_range(self.step, (x1, x2))
        if not m.valid:
            try:
                from iminuit.qtwidget import make_widget

            except ImportError:
                print("iminuit.qtwidget not installed")
            else:
                # Create the widget
                self.debug_fit = make_widget(m, m._visualize(None), {}, False, False)
                self.debug_fit.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
                self.debug_fit.destroyed.connect(lambda: self.on_fit_closed(self.step, m))
                self.debug_fit.show()
        else:
            # Clear the selector
            self.selector.clear()
            # Go to the next plot
            self.plot_next()

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
                self.scan.plot_eff(self.ax)
                self.canvas.draw()
        else:
            self.plot(self.step)


class PhexViewer(AGEDataViewer):

    def __init__(self,
        scan: EnergyScan,
        reference: pd.DataFrame,
        qnum: Sequence[str],
        energy_range: float,
        simulation: pd.DataFrame,
    ) -> None:
        self.scan = scan
        self.reference = reference
        self.qnum = qnum
        self.simulation = simulation
        self.assignments = {}
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
        self.y, self.yerr, self.x = self.scan.counts()
        self.ymax = np.max(self.y)
        # Prepare the assignment fit data for plotting
        self.x_fit = np.linspace(self.x[0], self.x[-1], len(self.x) * 100)
        self.y_fit = np.zeros_like(self.x_fit)
        self.yerr_fit = np.zeros_like(self.x_fit)
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
        in_range = (self.x_fit >= self.xlim[0]) & (self.x_fit <= self.xlim[1])
        x_fit = self.x_fit[in_range]
        y_fit = self.y_fit[in_range]
        yerr_fit = self.yerr_fit[in_range]
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
                text = f"{row[self.qnum[0]]}"
                for q in self.qnum[1:]:
                    text += f",{row[q]}"
                self.ax.text(row["E"] + 0.0002, self.ymax, text, ha="left",
                             va="top", rotation=90)
            # Plot the assignments
            self.ax.plot(x_fit, y_fit, color=ageplot.colors[1])
            self.ax.fill_between(x_fit, y_fit - yerr_fit, y_fit + yerr_fit,
                                color=ageplot.colors[1], alpha=0.5)
            self.ax.set_xlim(self.xlim)
            self.ax.set_ylim(top=self.ymax * 1.05)
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
        dialog = PhexDialog(self)
        if dialog.exec():
            exc = dialog.get_input()
            if self.simulation is not None:
                E = self.simulation.copy()
            else:
                E = self.reference.copy()
            for q, val in zip(self.qnum, exc):
                query_str = f"{q} == @val"
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
        # Import the fitting modules
        try:
            from iminuit import Minuit
            from iminuit.cost import LeastSquares
            from numba_stats import norm, bernstein

        except ImportError:
            raise ImportError("iminuit and numba-stats is required for fitting.")

        # Prepare the fit
        def model(x, *par):
            return par[0] * norm.pdf(x, *par[1:3]) + bernstein.density(x, par[3:], *xr)

        start = [np.max(y), (xr[0] + xr[1]) * 0.5, 0.01 * (xr[1] - xr[0]), 1, 1, 1, 1]
        limits = {
            "s": (0, None),
            "loc": (xr[0], xr[1]),
            "scale": (0.001 * (xr[1] - xr[0]), 0.5 * (xr[1] - xr[0])),
            "a0": (0, None),
            "a1": (0, None),
            "a2": (0, None),
            "a3": (0, None),
        }
        cost = LeastSquares(x, y, yerr, model)
        m = Minuit(cost, *start, name=limits.keys())
        # Set the limits
        for par in limits:
            m.limits[par] = limits[par]
        # Fix the higher order bernstein terms
        m.fixed["a1", "a2", "a3"] = True
        # Fit the data
        m.migrad()
        # Debug the fit if it failed
        if not m.valid:
            try:
                from iminuit.qtwidget import make_widget

            except ImportError:
                print("iminuit.qtwidget not installed")
            else:
                # Create the widget
                self.debug_fit = make_widget(m, m._visualize(None), {}, False, False)
                self.debug_fit.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
                self.debug_fit.destroyed.connect(lambda: self.on_fit_closed(m, model, xr))
                self.debug_fit.show()
        else:
            self.on_fit_closed(m, model, xr)

    def on_fit_closed(self, m, model, xr):
        self.debug_fit = None
        dialog = PhexDialog(self)
        if dialog.exec():
            exc = dialog.get_input()
            val = np.array(m.values)
            cov = np.array(m.covariance)
            self.assignments[tuple(exc)] = (val, cov)
            # Add the result to the fit data
            in_range = (self.x_fit >= xr[0]) & (self.x_fit <= xr[1])
            x_fit = self.x_fit[in_range]
            self.y_fit[in_range] = model(x_fit, *val)
            try:
                from jacobi import propagate

            except ImportError:
                warnings.warn("jacobi not installed, fit uncertainty is not plotted.")
                self.yerr_fit[in_range] = 0
            else:
                _, yerr_fit = propagate(lambda par: model(x_fit, *par), val, cov)
                self.yerr_fit[in_range] = np.sqrt(np.diag(yerr_fit))
            # Update the plot
            self.plot()
