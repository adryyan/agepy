"""Find the optimal focus and rotation."""

from __future__ import annotations

import numpy as np

from .scan import Scan
from .util import parse_calib

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray, ArrayLike
    from .anodes import PositionAnode


class FocusScan(Scan):
    """Scan over detector focus positions to find a good focus.

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

        # Initialize the result array
        self.fit = np.full(len(self.steps), [], dtype=object)
        self.loc = np.full(len(self.steps), [], dtype=object)
        self.theta = np.full(len(self.steps), [], dtype=object)
        self.chi2 = np.full(len(self.steps), [], dtype=object)
        self.dx = np.full(len(self.steps), [], dtype=object)
        self.dy = np.full(len(self.steps), [], dtype=object)

    @property
    def calib(self) -> NDArray:
        return self._calib

    @calib.setter
    def calib(self, calib: ArrayLike) -> None:
        self._calib = parse_calib(((0, 0), (1, 0)))

    @property
    def qeff(self) -> None:
        return None

    @qeff.setter
    def qeff(
        self, qeff: tuple[NDArray, NDArray, NDArray] | Scan | None
    ) -> None:
        self._qeff = None

    def interactive(
        self, bins: int | ArrayLike = 512, sig: str = "Voigt"
    ) -> int:
        """Interactively fit the spectra and find a good focus.

        Parameters
        ----------
        bins: int or array_like
            Bin number or edges between 0 and 1.
        sig: str
            The default signal model to use for fits. Can be changed
            in the interactive fit window.

        """
        from agepy.interactive import get_qapp
        from ._interactive_focussing import EvalFocus

        # Get the Qt application
        app = get_qapp()

        # Intialize the viewer
        mw = EvalFocus(self, bins, sig)
        mw.show()

        # Run the application
        return app.exec()
