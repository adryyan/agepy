from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from PySide6.QtWidgets import QPushButton
from PySide6.QtCore import Qt
# Import internal modules
from agepy.interactive import AGEDataViewer
from agepy import ageplot
# Import modules for type hinting
if TYPE_CHECKING:
    from agepy.spec.photons import Scan, QEffScan

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
        self.view_results = QPushButton("View Results")
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
                self.debug_fit.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
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
