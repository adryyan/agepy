"""Evaluate quantum efficiency measurements."""

from __future__ import annotations

import warnings
import pickle
import numpy as np
import matplotlib.pyplot as plt

from .scan import Scan
from .util import parse_calib

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray, ArrayLike
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from .anodes import PositionAnode


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

    @property
    def calib(self) -> NDArray:
        return self._calib

    @calib.setter
    def calib(self, calib: ArrayLike) -> None:
        if calib != ((0, 0), (1, 0)):
            wrnmsg = "Cannot set custom calib for QEffScan."
            warnings.warn(wrnmsg, stacklevel=1)

        self._calib = parse_calib(((0, 0), (1, 0)))

    @property
    def qeff(self) -> tuple[NDArray, NDArray, NDArray] | None:
        y, yerr, x = [], [], []

        # Append the fit results
        for fit in self.fit:
            if fit is None:
                continue

            y.append(fit.value("s"))
            yerr.append(fit.error("s"))
            x.append(fit.value("loc"))

        # Return None if fits were performed yet
        if len(y):
            return None

        # Convert to numpy arrays
        y = np.asarray(y, dtype=np.float64)
        yerr = np.asarray(yerr, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)

        # Normalize the values
        ymax = np.max(y)

        # Sort the values
        inds = np.argsort(x)

        return y[inds] / ymax, yerr[inds] / ymax, x[inds]

    @qeff.setter
    def qeff(
        self, qeff: tuple[NDArray, NDArray, NDArray] | Scan | None
    ) -> None:
        if qeff is not None:
            wrnmsg = "Cannot manually set qeff for QEffScan."
            warnings.warn(wrnmsg, stacklevel=1)

        self._qeff = None

    def interpolate(
        self,
        x: ArrayLike,
        mc_samples: int = 10000,
    ) -> tuple[NDArray, NDArray]:
        # Get the fit values
        qeff = self.qeff

        if qeff is not None:
            py, pyerr, px = qeff

        else:
            errmsg = "Quantum efficiency is not evaluated."
            raise ValueError(errmsg)

        # Generate samples
        rng = np.random.default_rng()
        y_samples = rng.normal(loc=py, scale=pyerr, size=(mc_samples, len(py)))
        eff_samples = np.stack(
            [np.interp(x, px, y, left=0, right=0) for y in y_samples], axis=0
        )

        # Calculate the mean and standard deviation
        eff = np.mean(eff_samples, axis=0, keepdims=True)
        err = np.std(eff_samples, axis=0, ddof=1, mean=eff)
        eff = eff.flatten()

        return eff, err

    def interactive(
        self, bins: int | ArrayLike = 512, sig="Voigt", bkg="None"
    ) -> int:
        """Interactively evaluate the quantum efficiencies by fitting
        peaks in the spectra.

        Parameters
        ----------
        bins: int or array_like
            Bin number or edges between 0 and 1.
        sig: str
            The default signal model to use for fits. Can be changed
            in the interactive fit window.
        bkg: str
            The default background model to use for fits. Can be
            changed in the interactive fit window.

        """
        from agepy.interactive import get_qapp
        from ._interactive_qeff import EvalQEff

        # Get the Qt application
        app = get_qapp()

        # Intialize the viewer
        mw = EvalQEff(self, bins, sig, bkg)
        mw.show()

        # Run the application
        return app.exec()

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

    def save_qeff(self, filepath: str) -> None:
        """
        Save the evaluated quantum efficiencies.

        Parameters
        ----------
        filepath: str
            Path to the file where the qeff will be saved.

        """
        qeff = self.qeff

        if qeff is None:
            errmsg = "Quantum efficiency is not evaluated."
            raise ValueError(errmsg)

        with open(filepath, "wb") as f:
            pickle.dump(qeff, f)
