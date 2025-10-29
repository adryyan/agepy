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
    from numpy.typing import NDArray
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class QEff:
    xval: NDArray
    xerr: NDArray
    yval: NDArray
    yerr: NDArray
    cache: tuple[NDArray, NDArray] = field(default=None, init=False)

    @contextmanager
    def cache_eff(self, xe: NDArray, n: int = 10000):
        """Cache efficiencies for a given set of x edges."""
        cache = self.eff(xe, n=n)
        object.__setattr__(self, "cache", cache)
        try:
            yield

        finally:
            object.__setattr__(self, "cache", None)

    def eff(self, xe: NDArray, n: int = 10000) -> tuple[NDArray, NDArray]:
        """Interpolated quantum efficiencies with uncertainties
        propagated with Monte Carlo error propagation.

        Parameters
        ----------
        xe: np.ndarray, shape(N,)
            Detector binning. The interpolation is evaluated at the
            bin centers.
        n: int, optional
            Number of Monte Carlo samples to generate.

        Returns
        -------
        eff: np.ndarray, shape(N-1,)
            The interpolated efficiencies.
        err: np.ndarray, shape(N-1,)
            The corresponding uncertainties.

        """
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
        eff_samples = _generate_eff_samples(xc, x_samples, y_samples, n)

        # Calculate the standard deviation
        err = np.std(eff_samples, axis=0, ddof=1)

        return eff, err


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


def plot_qeff(
    qeff: QEff,
    ax: Axes | None = None,
    color: str = "k",
    label: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot the calculated detector efficiencies.

    Parameters
    ----------
    qeff: Qeff,
        Evaluated quantum efficiency data.
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
    xe = np.histogram([], bins=1024, range=(0, 1))
    xc = (xe[1:] + xe[:-1]) * 0.5
    eff, err = qeff.eff(xe)

    ax.errorbar(
        qeff.xval,
        qeff.yval,
        yerr=qeff.yerr,
        xerr=qeff.xerr,
        fmt="s",
        color=color,
        label=label,
    )

    # Fix the ylim
    ylim = ax.get_ylim()
    ax.set_ylim(ylim)

    # Plot the interpolated values
    ax.plot(xc, eff, color=color, linestyle="-")
    ax.fill_between(xc, eff - err, eff + err, color=color, alpha=0.3)

    # Set ylim back to auto
    ax.set_ylim(auto=True)

    # Set the labels
    ax.set_xlabel("Detector Position [arb. u.]")
    ax.set_ylabel("Efficiency [arb. u.]")
    ax.set_xlim(0, 1)
    ax.set_title("Measured Lateral Quantum Efficiency")

    return fig, ax


@njit(parallel=True, fastmath=True)
def _generate_eff_samples(x: NDArray, xs: NDArray, ys: NDArray) -> NDArray:
    effs = np.zeros((ys.size, x.size), dtype=np.float64)

    for i in prange(ys.size):
        effs[i] = np.interp(x, xs[i], ys[i])

    return effs
