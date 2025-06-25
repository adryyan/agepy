"""Find good focussing of the detector."""

from __future__ import annotations

import matplotlib.pyplot as plt

from .qeff import QEffScan

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class FocusScan(QEffScan):
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

    @property
    def qeff(self) -> tuple[NDArray, NDArray, NDArray] | None:
        return None

    @qeff.setter
    def qeff(self, qeff: tuple[NDArray, NDArray, NDArray] | None) -> None:
        self._qeff = None

    def plot(
        self,
        par: str,
        ax: Axes | None = None,
        color: str = "k",
        label: str | None = None,
    ) -> tuple[Figure, Axes]:
        """Plot the calculated detector efficiencies.

        Parameters
        ----------
        par: str
            The parameter to plot.
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

        # Get the fit values
        val, val_err, loc, loc_err = [], [], [], []

        for fit in self.fit:
            loc.append(fit.value("loc"))
            loc_err.append(fit.error("loc"))

            if par == "chi2":
                val.append(fit.chi2)
                val_err.append(0)

            else:
                val.append(fit.value(par))
                val_err.append(fit.error(par))

        ax.errorbar(
            loc,
            val,
            yerr=val_err,
            xerr=loc_err,
            fmt="s",
            color=color,
            label=label,
        )

        # Set the labels
        ax.set_xlabel("Detector Position [arb. u.]")
        ax.set_ylabel(par)
        ax.set_xlim(0, 1)

        return fig, ax
