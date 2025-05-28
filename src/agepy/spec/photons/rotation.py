"""Find the optimal detector rotation."""

from __future__ import annotations

import numpy as np

from .scan import Scan

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Union, Tuple, Sequence
    from numpy.typing import NDArray
    from agepy.spec.photons.anodes import PositionAnode


class RotationScan(Scan):
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

    Attributes
    ----------
    spectra: np.ndarray
        Array of the loaded Spectrum objects.
    steps: np.ndarray
        Array of the scan variable values.

    """

    def __init__(
        self,
        data_files: Sequence[str],
        anode: PositionAnode,
        raw: str = "dld_rd#raw",
        time_per_step: Union[int, Sequence[int]] = None,
    ) -> None:
        # Force the roi to cover the full detector
        roi = ((0, 1), (0, 1))

        # Load and process data
        super().__init__(data_files, anode, None, raw, time_per_step, roi)

        # Set attributes
        self._angle = None
        self._offset = None

    @property
    def calib(self) -> None:
        return None

    @calib.setter
    def calib(self, value) -> None:
        raise AttributeError("Calibration can not be set for QEffScan.")

    @property
    def qeff(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return None

    @qeff.setter
    def qeff(self, value) -> None:
        raise AttributeError("Quantum efficiency can not be set for QEffScan.")

    @property
    def rotation(self) -> NDArray:
        return self._angle

    @property
    def offset(self) -> NDArray:
        return self._offset

    def interactive(self, angle: float, offset: Tuple[float, float]) -> int:
        """Plot the spectra in an interactive window."""
        from agepy.interactive import get_qapp
        from agepy.spec.interactive.photons_rot import EvalRot

        # Get the Qt application
        app = get_qapp()

        # Intialize the viewer
        mw = EvalRot(self, angle, offset)
        mw.show()

        # Run the application
        return app.exec()
