"""Evaluate quantum efficiency measurements."""

from __future__ import annotations

import pickle
import numpy as np
import matplotlib.pyplot as plt

from .scan import Scan

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Union, Tuple, Literal, Sequence
    from numpy.typing import NDArray
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from agepy.spec.photons.anodes import PositionAnode

    type RegionOfInterest = Tuple[Tuple[float, float], Tuple[float, float]]
    type QuantumEfficiency = Tuple[NDArray, NDArray, NDArray]
    type WavelengthCalib = Tuple[Tuple[float, float], Tuple[float, float]]
    type ErrorPropagation = Literal["montecarlo", "none"]


class QEffScan(Scan):
    """Scan over grating positions with a spectrum for each step.

    Parameters
    ----------
    data_files: Sequence[str]
        List of data files to be processed.
    anode: PositionAnode
        Anode object from `agepy.spec.photons`.
    raw: str, optional
        Path to the raw data in the data files. Default:
        "dld_rd#raw/0".
    time_per_step: int, optional
        Time per step in the scan. Default: None.
    **norm
        Path to normalization parameters in the h5 data files.

    Attributes
    ----------
    spectra: np.ndarray
        Array of the loaded Spectrum objects.
    energies: np.ndarray
        Array of the scan variable values.

    """

    def __init__(
        self,
        data_files: Sequence[str],
        anode: PositionAnode,
        raw: str = "dld_rd#raw",
        time_per_step: Union[int, Sequence[int]] = None,
        roi: dict = None,
        **norm,
    ) -> None:
        # Force the x roi to cover the full detector
        if roi is not None:
            roi[0][0] = 0
            roi[0][1] = 1

        # Load and process data
        super().__init__(
            data_files, anode, None, raw, time_per_step, roi, **norm
        )

        # Initialize the result arrays
        n = len(self.steps)
        self._py = np.zeros(n, dtype=np.float64)
        self._pyerr = np.zeros(n, dtype=np.float64)
        self._px = np.zeros(n, dtype=np.float64)
        self._pxerr = np.zeros(n, dtype=np.float64)

    @property
    def calib(self) -> None:
        return None

    @calib.setter
    def calib(self, value) -> None:
        raise AttributeError("Calibration can not be set for QEffScan.")

    @property
    def qeff(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Remove zeros
        inds = np.argwhere(self._py > 0).flatten()

        # Check if there are any non-zero values
        if len(inds) == 0:
            raise ValueError("Quantum Efficiency is not evaluated.")

        px = self._px[inds]
        py = self._py[inds]
        pyerr = self._pyerr[inds]

        # Normalize the values
        ymax = np.max(py)

        # Sort the values
        inds = np.argsort(px)

        return py[inds] / ymax, pyerr[inds] / ymax, px[inds]

    @qeff.setter
    def qeff(self, value) -> None:
        raise AttributeError("Quantum efficiency can not be set for QEffScan.")

    def interpolate(
        self,
        x: np.ndarray,
        mc_samples: int = 10000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Get the fit values
        qeff = self.qeff
        if qeff is not None:
            py, pyerr, px = qeff

        else:
            raise ValueError("Quantum efficiency is not evaluated.")

        # Interpolate the efficiencies
        def interp(y):
            return np.interp(x, px, y)

        # Generate samples
        rng = np.random.default_rng()
        y_samples = rng.normal(loc=py, scale=pyerr, size=(mc_samples, len(py)))
        eff_samples = np.stack([interp(y) for y in y_samples], axis=0)

        # Calculate the mean and standard deviation
        eff = np.mean(eff_samples, axis=0)
        err = np.std(eff_samples, axis=0)

        # Set values outside the interpolation range to 0
        eff[x < px[0]] = 0
        eff[x > px[-1]] = 0
        err[x < px[0]] = 0
        err[x > px[-1]] = 0
        return eff, err

    def interactive(self, bins: int = 512, sig="Voigt", bkg="None") -> int:
        """Plot the spectra in an interactive window."""
        from agepy.interactive import get_qapp
        from agepy.spec.interactive.photons_qeff import EvalQEff

        # Create the edges
        edges = np.histogram([], bins=bins, range=(0, 1))[1]

        # Get the Qt application
        app = get_qapp()

        # Intialize the viewer
        mw = EvalQEff(self, edges, sig, bkg)
        mw.show()

        # Run the application
        return app.exec()

    def plot(
        self, ax: Axes = None, color: str = "k", label: str = None
    ) -> Tuple[Figure, Axes]:
        """Plot the calculated detector efficiencies."""
        # Create the figure and axis
        if ax is None:
            fig, ax = plt.subplots()

        else:
            fig = ax.get_figure()

        # Plot fit values
        qeff = self.qeff
        if qeff is not None:
            y, yerr, x = qeff

        else:
            raise ValueError("Quantum efficiency is not evaluated.")

        ax.errorbar(x, y, yerr=yerr, fmt="s", color=color, label=label)

        # Fix the ylim
        ylim = ax.get_ylim()
        ax.set_ylim(ylim)

        # Plot the interpolated values
        x = np.linspace(0, 1, 1000)
        eff, err = self.interpolate(x)
        ax.plot(x, eff, color=color, linestyle="-")
        ax.fill_between(x, eff - err, eff + err, color=color, alpha=0.3)

        # Set ylim back to auto
        ax.set_ylim(auto=True)

        # Set the labels
        ax.set_xlabel("Detector Position [arb. u.]")
        ax.set_ylabel("Efficiency [arb. u.]")
        ax.set_xlim(0, 1)
        ax.set_title("Measured Lateral Quantum Efficiency")

        return fig, ax

    def save(self, filepath: str) -> None:
        """
        Save the evaluated quantum efficiencies.

        Parameters
        ----------
        filepath: str
            Path to the file where the qeff will be saved.

        """
        with open(filepath, "wb") as f:
            pickle.dump(self.qeff, f)

    @staticmethod
    def load(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load the evaluated quantum efficiencies.

        Parameters
        ----------
        filepath: str
            Path to the file where the qeff is saved.

        """
        with open(filepath, "rb") as f:
            return pickle.load(f)
