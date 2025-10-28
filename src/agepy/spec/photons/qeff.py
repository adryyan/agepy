"""Evaluate quantum efficiency measurements."""

from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import contextmanager
import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt

from .scan import Scan

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray, ArrayLike
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from .anodes import PositionAnode


@dataclass(frozen=True)
class QEff:
    xval: NDArray
    xerr: NDArray
    yval: NDArray
    yerr: NDArray
    cache: tuple[NDArray, NDArray] = field(default=None, init=False)

    @contextmanager
    def cache_eff(self, xe: NDArray, mc_samples: int = 10000):
        """Cache efficiencies for a given set of x edges."""
        cache = self.eff(xe, mc_samples=mc_samples)
        object.__setattr__(self, "cache", cache)
        try:
            yield

        finally:
            object.__setattr__(self, "cache", None)

    def eff(self, xe: NDArray, n: int = 10000) -> tuple[NDArray, NDArray]:
        # Return cached efficiencies (only with context)
        if self.cache is not None:
            return self.cache

        # Get the bin centers
        xc = (xe[1:] + xe[:-1]) * 0.5

        # Interpolate and evaluate at the given bin centers
        eff = np.interp(xc, self.xval, self.yval, left=np.nan, right=np.nan)

        # Generate random samples for x and y
        rng = np.random.default_rng()
        y_samples = rng.normal(
            loc=self.yval, scale=self.yerr, size=(n, self.yval.size)
        )
        x_samples = rng.normal(
            loc=self.xval, scale=self.xerr, size=(n, self.xval.size)
        )

        # Generate the efficiency samples
        eff_samples = self.generate_eff_samples(xc, x_samples, y_samples, n)

        # Calculate the standard deviation
        err = np.std(eff_samples, axis=0, ddof=1)

        return eff, err

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def generate_eff_samples(
        x: NDArray, x_samples: NDArray, y_samples: NDArray, n: int
    ) -> NDArray:
        eff_samples = np.zeros((n, x.size), dtype=np.float64)

        for i in prange(n):
            eff_samples[i] = np.interp(x, x_samples[i], y_samples[i])

        return eff_samples


def eval_qeff(
    scan: Scan, bins: int = 512, sig: str = "Voigt", bkg: str = "Constant"
) -> QEff | None:
    """Interactively evaluate the quantum efficiencies by fitting
    peaks in the spectra.

    Parameters
    ----------
    scan: Scan
        Quantum efficiency measurement.
    bins: int or array_like
        Bin number or edges between 0 and 1.
    sig: str
        The default signal model to use for fits. Can be changed
        in the interactive fit window.
    bkg: str
        The default background model to use for fits. Can be
        changed in the interactive fit window.

    """
    from agepy.qt import get_qtapp
    from .qeff_widget import EvalQEff

    # Get the Qt application
    app = get_qtapp()

    # Intialize the viewer
    mw = EvalQEff(scan, bins, sig, bkg)
    mw.show()

    # Run the application
    app.exec()

    yval, yerr, xval, xerr = [], [], [], []

    # Append the fit results
    for fit in mw.fit:
        if fit is None:
            continue

        yval.append(fit.value("n"))
        yerr.append(fit.error("n"))
        xval.append(fit.value("loc"))
        xerr.append(fit.error("loc"))

    # Return None if only one or no fits were performed yet
    if len(yval) < 2:
        return None

    # Convert to numpy arrays
    yval = np.asarray(yval, dtype=np.float64)
    yerr = np.asarray(yerr, dtype=np.float64)
    xval = np.asarray(xval, dtype=np.float64)
    xerr = np.asarray(xerr, dtype=np.float64)

    # Normalize the values
    ymax = np.max(yval)

    # Sort the values
    inds = np.argsort(xval)

    return QEff(
        yval[inds] / ymax,
        yerr[inds] / ymax,
        xval,
        xerr,
    )


class QEffScan(Scan):
    """Scan over grating positions with a spectrum for each step.

    Parameters
    ----------
    data_files: array_like
        List of data files (str) to be processed.
    anode: PositionAnode
        Anode object to process the raw data.
    raw: str, optional
        Path to the raw data in the data files.
    time_per_step: int, optional
        Time per step in the scan.
    roi: array_like, shape (2,2), optional
        Region of interest for the detector in the form
        `((xmin, xmax), (ymin, ymax))`.
    **normalize: str
        Path to additional normalization parameters as keyword
        arguments like the upstream intensity or target density.

    Attributes
    ----------
    spectra: np.ndarray, shape (N,)
        Array of the loaded Spectrum objects.
    steps: np.ndarray, shape (N,)
        Array of the scan variable values.
    m_id: np.ndarray, shape (N,)
        Array of the measurement numbers.
    roi: np.ndarray, shape (2,2)
        Region of interest for the detector.
    qeff: [np.ndarray, np.ndarray, np.ndarray] or None
        Detector efficiencies in the form `(values, errors, x)`
        with shapes (M,).

    """

    def __init__(
        self,
        data_files: str | ArrayLike,
        anode: PositionAnode,
        raw: str = "dld_rd#raw",
        time_per_step: int | ArrayLike | None = None,
        roi: ArrayLike = ((0, 1), (0, 1)),
        **normalize: str,
    ) -> None:
        # Load and process data
        super().__init__(
            data_files,
            anode,
            scan_var=None,
            raw=raw,
            time_per_step=time_per_step,
            roi=roi,
            **normalize,
        )

        # Force the x roi to cover the full detector
        self.roi[0, 0] = 0
        self.roi[0, 1] = 1

        # Initialize the result arrays
        self.fit = np.full(len(self.steps), None, dtype=object)

    def plot(
        self,
        ax: Axes | None = None,
        color: str = "k",
        label: str | None = None,
    ) -> tuple[Figure, Axes]:
        """Plot the calculated detector efficiencies.

        Parameters
        ----------
        ax: Axes, optional
            A matplotlib axes to draw on.
        color: str, optional
            A color to use for the efficiencies.
        label: str, optional
            A label for the plotted data.

        Returns
        -------
        fig: Figure
            The matplotlib figure.
        ax: Axes
            The matplotlib axes.

        """
        # Create the figure and axis
        if ax is None:
            fig, ax = plt.subplots()

        else:
            fig = ax.get_figure()

        # Get the interpolated efficiencies
        x_interp = np.linspace(0, 1, 1000)
        eff, err = self.interpolate(x_interp)

        # Get the fit values
        y, yerr, x = self.qeff

        ax.errorbar(x, y, yerr=yerr, fmt="s", color=color, label=label)

        # Fix the ylim
        ylim = ax.get_ylim()
        ax.set_ylim(ylim)

        # Plot the interpolated values
        ax.plot(x_interp, eff, color=color, linestyle="-")
        ax.fill_between(x_interp, eff - err, eff + err, color=color, alpha=0.3)

        # Set ylim back to auto
        ax.set_ylim(auto=True)

        # Set the labels
        ax.set_xlabel("Detector Position [arb. u.]")
        ax.set_ylabel("Efficiency [arb. u.]")
        ax.set_xlim(0, 1)
        ax.set_title("Measured Lateral Quantum Efficiency")

        return fig, ax
