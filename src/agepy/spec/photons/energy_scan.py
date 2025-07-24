"""Load and process exciting-photon energy scans of H2."""

from __future__ import annotations

import warnings
import pickle
import numpy as np
import pandas as pd

from .scan import Scan

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray, ArrayLike
    from .anodes import PositionAnode
    from .spectrum import Spectrum
    from agepy.spec.fit_models import FitModel1d


class EnergyScan(Scan):
    """Scan over exciting-photon energies with a spectrum for each
    energy step.

    Parameters
    ----------
    data_files: array_like
        List of data files (str) to be processed.
    anode: PositionAnode
        Anode object to process the raw data.
    energies: str, optional
        Path to the step values in the data files. If `None`,
        the keys are used as the values.
    raw: str, optional
        Path to the raw data in the data files.
    time_per_step: int, optional
        Time per step in the scan.
    roi: array_like, shape (2,2), optional
        Region of interest for the detector in the form
        `((xmin, xmax), (ymin, ymax))`.
    qeff: [np.ndarray, np.ndarray, np.ndarray], optional
        Detector efficiencies in the form `(values, errors, x)`
        with shapes (M,).
    bkg: Spectrum, optional
        Background spectrum (dark counts) to be subtracted from
        the spectra.
    calib: array_like, shape (2,2), optional
        Wavelength calibration parameters in the form
        `((a0, err), (a1, err))`, where `a0` and `a1`
        correspond to $\\lambda = a_1 x + a_0$ and `err` to the
        respective uncertainties.
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
    bkg: Spectrum or None
        Background spectrum (dark counts) to be subtracted.
    calib: np.ndarray, shape (2,2)
        Wavelength calibration parameters in the form
        `((a0, err), (a1, err))`, where `a0` and `a1`
        correspond to $\\lambda = a_1 x + a_0$ and `err` to the
        respective uncertainties.

    """

    def __init__(
        self,
        data_files: str | ArrayLike,
        anode: PositionAnode,
        energies: str | None = None,
        raw: str = "dld_rd#raw",
        time_per_step: int | ArrayLike | None = None,
        roi: ArrayLike = ((0, 1), (0, 1)),
        qeff: tuple[NDArray, NDArray, NDArray] | None = None,
        bkg: Spectrum | None = None,
        calib: ArrayLike = ((0, 0), (1, 0)),
        energy_uncertainty: int | float | ArrayLike = 0,
        **normalize: str,
    ) -> None:
        # Load and process data
        super().__init__(
            data_files,
            anode,
            scan_var=energies,
            raw=raw,
            time_per_step=time_per_step,
            roi=roi,
            qeff=qeff,
            bkg=bkg,
            calib=calib,
            **normalize,
        )

        # Set attributes
        self.energy_uncertainty = energy_uncertainty

        # Create DataFrames for the assignments
        col = [
            "J",
            "Elp",
            "vp",
            "Jp",
            "phex_fit",
            "exc_energy",
            "Elpp",
            "vpp",
            "Jpp",
            "phem_fit",
            "emi_energy",
        ]

        self._bound = pd.DataFrame(columns=col)

    @property
    def energies(self) -> NDArray:
        return self._steps

    @energies.setter
    def energies(self, value: NDArray) -> None:
        self.steps = value

    @property
    def energy_uncertainty(self) -> NDArray:
        return self._energy_uncertainty

    @energy_uncertainty.setter
    def energy_uncertainty(self, value: int | float | ArrayLike) -> None:
        if isinstance(value, (int, float)):
            self._energy_uncertainty = np.full(
                len(self._steps), value, dtype=np.float64
            )

        else:
            value = np.array(value, dtype=np.float64)

            if value.shape == self.steps.shape:
                self._energy_uncertainty = value

            else:
                errmsg = "energy_uncertainty must have same length as steps."
                raise ValueError(errmsg)

    def get_assignment(self, **qnumbers):
        df = self._bound
        for qn, val in qnumbers.items():  # noqa F841
            if df.empty:
                return None, None

            df = df.query(f"{qn} == @val")

        if df.empty:
            return None, None

        df = df.iloc[0]

        return df["phex_fit"], df["phem_fit"]

    def set_assignment(
        self,
        phex_fit: FitModel1d | None,
        phem_fit: FitModel1d | None,
        **qnumbers: int | str,
    ) -> None:
        # Find the index where to save the assignment
        df = self._bound
        for qn, val in qnumbers.items():  # noqa
            if df.empty:
                break

            df = df.query(f"{qn} == @val")

        if df.empty:
            if phex_fit is None or phem_fit is None:
                return

            idx = self._bound.index.max()
            if np.isnan(idx):
                idx = 0

            else:
                idx += 1

        else:
            idx = df.index[0]

            if phex_fit is None or phem_fit is None:
                self._bound.drop(index=idx)
                return

        row = qnumbers.copy()
        row["phex_fit"] = phex_fit
        row["exc_energy"] = phex_fit.value("loc")
        row["phem_fit"] = phem_fit
        row["emi_energy"] = phem_fit.value("loc")

        self._bound.loc[idx] = row

    def remove_steps(
        self,
        measurement_id: str,
        steps: ArrayLike,
    ) -> None:
        """Remove the specified steps of a measurement from the scan.

        Parameters
        ----------
        measurement_id: str
            Measurement number (metro) to remove the steps from.
        steps: array_like
            List of step values to remove.

        """
        # Select the spectra corresponding to the measurement number
        inds = np.nonzero(self.m_id == measurement_id)[0]

        # Check if the measurement number exists
        if len(inds) == 0:
            errmsg = f"No spectra found for m_id {measurement_id}."
            raise ValueError(errmsg)

        steps = np.array(steps, dtype=np.float64)

        # Select the steps to remove
        step_inds = []
        for step in steps:
            # Select the index of the closest step value
            step_inds.append(np.argsort(np.abs(self.steps[inds] - step))[0])

        # Check if the steps exist
        if len(step_inds) == 0:
            errmsg = f"Steps not found in {measurement_id}."
            raise ValueError(errmsg)

        inds = inds[step_inds]

        # Remove the steps
        self._steps = np.delete(self.steps, inds)
        self._spectra = np.delete(self.spectra, inds)
        self._m_id = np.delete(self.m_id, inds)

        # Remove the uncertainties
        self._energy_uncertainty = np.delete(self._energy_uncertainty, inds)

    def select_by_phex(
        self,
        J: int,
        Elp: str,
        vp: int,
        Jp: int,
        n_std: int = 1,
        ignore_overlap: bool = False,
    ) -> NDArray:
        # Parse the quantum numbers
        idx = f"{J},{Elp},{vp},{Jp}"

        # Find the phex assignment
        if idx not in self._phex.index:
            errmsg = "phex assignment not found."
            raise ValueError(errmsg)

        phex = self._phex.loc[idx]

        # Get the fit results
        fit = phex["fit"]

        # Select energy steps within n_std standard deviations of the mean
        step_idx = np.argwhere(
            np.abs(self.steps - fit.val[1]) < fit.val[2] * n_std
        ).flatten()

        # Check if steps were found
        if len(step_idx) == 0 or ignore_overlap:
            return step_idx

        # Define energy range
        e_range = (  # noqa F841
            fit.val[1] - fit.val[2] * n_std,
            fit.val[1] + fit.val[2] * n_std,
        )

        # Check if multiple phex assignments overlap
        overlap = self._phex.query(
            "exc_energy > @e_range[0] and exc_energy < @e_range[1]"
        )

        for row in overlap.itertuples():
            if row["Index"] == idx:
                continue

            overlap_idx = np.argwhere(
                np.abs(self.steps - row["fit"].val[1]) < row["fit"].val[2]
            ).flatten()

            # Remove the overlapping steps
            overlap_idx = np.setdiff1d(step_idx, overlap_idx)

            # Check if steps remain
            if len(overlap_idx) == 0:
                wrnmsg = "No steps found without overlap"
                warnings.warn(wrnmsg, stacklevel=1)

            else:
                step_idx = overlap_idx

        return step_idx

    def assign_phexphem(
        self,
        reference: pd.DataFrame,
        phem_calib: tuple[float, float],
    ) -> int:
        """Interactively assign 2d photon-excitation
        photon-emission peaks in the PhexPhem map.

        Parameters
        ----------
        reference: pd.DataFrame
            Simulation of bound PhexPhem transitions.

        """
        from agepy.qt import get_qtapp
        from .qt_phexphem import PhexPhemViewer

        # Get the Qt application
        app = get_qtapp()

        # Intialize the viewer
        mw = PhexPhemViewer(self, reference, phem_calib)
        mw.show()

        # Run the application
        return app.exec()

    def save_phex(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self._bound, f)

    def load_phex(self, path: str) -> None:
        with open(path, "rb") as f:
            self._bound = pickle.load(f)

    def phexphem(
        self,
        bins: int | ArrayLike = 512,
        qeff: bool = True,
        bkg: bool = True,
        calib: bool = True,
        mc_errors: bool = True,
        mc_samples: int = 10000,
        mc_seed: int | None = None,
        mc_spectrum: bool = False,
    ) -> NDArray:
        # Prepare the x edges
        xe = np.arange(11.0005, 18, 0.001)

        # Parse the y edges
        a = self.calib if calib else ((0, 0), (1, 0))
        ran = (a[0][0], a[0][0] + a[1][0])
        ran = (min(ran), max(ran))
        ye = np.histogram([], bins=bins, range=ran)[1]

        # Create an empty map
        hist = np.zeros((len(xe) - 1, len(ye) - 1))
        errors = np.zeros((len(xe) - 1, len(ye) - 1))

        # Fill the map
        weights = np.histogram(self.steps, bins=xe)[0].T
        inds = np.digitize(self.steps, xe[1:-1])

        for idx in range(len(self.steps)):
            spec, err = self.spectrum_at(
                idx,
                bins=bins,
                qeff=qeff,
                bkg=bkg,
                calib=calib,
                mc_errors=mc_errors,
                mc_samples=mc_samples,
                mc_seed=mc_seed,
                mc_spectrum=mc_spectrum,
            )[:2]

            hist[inds[idx], :] += spec
            errors[inds[idx], :] += err**2

        # Normalize the map
        weights[weights == 0] = 1
        hist = np.divide(hist, weights[:, np.newaxis])
        errors = np.sqrt(np.divide(errors, weights[:, np.newaxis] ** 2))

        return hist, errors, xe, ye
