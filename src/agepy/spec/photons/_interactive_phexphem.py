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

try:
    import pyqtgraph as pg
    from pyqtgraph import PlotWidget, ImageView, GraphicsLayoutWidget
    from pyqtgraph.Qt import QtCore as pgQtCore

except ImportError as e:
    errmsg = "pyqtgraph required for interactive plotting."
    raise ImportError(errmsg) from e

import numpy as np
from jacobi import propagate
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

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


class PhexPhemViewer(MainWindow):
    def __init__(
        self,
        scan: EnergyScan,
        reference: pd.DataFrame,
        phem_calib: tuple[float, float],
    ) -> None:
        self.scan = scan
        self.reference = reference

        # Prepare the data
        self.map, self.err, self.xe, self.ye = scan.phexphem(
            calib=False,
            mc_errors=False,
        )

        # Set up a meshgrid for plotting
        self.xm, self.ym = np.meshgrid(self.xe, self.ye)

        # Prepare the plotting
        self.fig = plt.figure(figsize=(6.4, 6.4), clear=True)

        gs = gridspec.GridSpec(
            2,
            2,
            width_ratios=[3, 1],
            height_ratios=[1, 3],
            wspace=0.05,
            hspace=0.05,
        )

        # 2d detector image is subplot 2: lower left
        self.ax_map = plt.subplot(gs[2])

        # x projection is subplot 0: upper left
        self.ax_exc = plt.subplot(gs[0], sharex=self.ax_map)

        # y projection is subplot 3: lower right
        self.ax_emi = plt.subplot(gs[3], sharey=self.ax_map)

        # colorbar is subplot 1: upper right
        ax_cb = plt.subplot(gs[1])
        ax_cb.axis("off")
        self.ax_cb = ax_cb.inset_axes([0.0, 0.0, 0.25, 1.0])

        # Remove unnecessary x and y tick labels
        self.ax_exc.tick_params(axis="both", labelbottom=False)
        self.ax_emi.tick_params(axis="both", labelleft=False)

        # Remove grid from the map and colorbar
        self.ax_map.grid(False)
        self.ax_cb.grid(False)

        # Flip y axis
        self.ax_map.set_ylim(0.9, 0.1)

        # Set labels
        self.ax_map.set_xlabel("Exciting Photon Energy [eV]")
        self.ax_map.set_ylabel("Detector Position [arb. u.]")

        # Get the color for the projections
        self.color = plt.get_cmap("viridis")(0)

        # Set up the main window
        super().__init__(width=960, height=1080)

        self.add_plot(self.fig, self.ax_map, width=960, height=960)
        self.add_toolbar()

        self.plot()

    def plot(self) -> None:
        hist = 1 - np.exp(-20 * (self.map / self.map.max()))
        # Plot the map
        pcm = self.ax_map.pcolormesh(
            self.xm, self.ym, hist, cmap="viridis", rasterized=True
        )

        # Create a colorbar
        self.fig.colorbar(pcm, cax=self.ax_cb)

        # Project the map onto the x and y axes
        x_proj = np.sum(self.map, axis=0)
        y_proj = np.sum(self.map, axis=1)

        # Plot the x and y projections
        self.ax_exc.stairs(x_proj, self.xe, color=self.color)
        self.ax_emi.stairs(
            y_proj, self.ye, color=self.color, orientation="horizontal"
        )

        # Remove the first tick label of the x and y projection
        plt.setp(self.ax_exc.get_yticklabels()[0], visible=False)
        plt.setp(self.ax_emi.get_xticklabels()[0], visible=False)

        self.canvas.draw_idle()
