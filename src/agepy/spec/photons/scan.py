"""Load fluorescence spectra from scans."""

from __future__ import annotations

import os
import pickle
import numpy as np
import h5py

from .spectrum import Spectrum
from .util import parse_roi, parse_qeff, parse_calib

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal
    from numpy.typing import NDArray, ArrayLike
    from agepy.spec.photons.anodes import PositionAnode


class BaseScan:
    def __init__(
        self,
        data_files: str | ArrayLike,
        anode: PositionAnode,
        scan_var: str | None = None,
        raw: str = "dld_rd#raw",
        time_per_step: int | ArrayLike | None = None,
        roi: ArrayLike = ((0, 1), (0, 1)),
        **normalize: str,
    ) -> None:
        # Intialize lists for the spectra and steps
        self._spectra = []
        self._steps = []

        # Keep track of the measurement number
        self._m_id = []

        # Force data_files to be a list
        if isinstance(data_files, str):
            data_files = [data_files]

        data_files = np.asarray(data_files, dtype=np.dtypes.StringDType())

        # Create a list of time_per_step values if only one is provided
        if isinstance(time_per_step, int):
            time_per_step = [time_per_step] * len(data_files)

        if time_per_step is not None:
            time_per_step = np.asarray(time_per_step, dtype=np.int64)

            # Check if the length of time_per_step matches the number of
            # data files
            if len(time_per_step) != len(data_files):
                errmsg = "time_per_step must have same length as data_files."
                raise ValueError(errmsg)

        else:
            # Use 1s as the time per step
            time_per_step = np.ones_like(data_files, dtype=np.int64)

        # Load the spectra from the data files
        for f, t in zip(data_files, time_per_step):
            spectra, steps = self._load_spectra(
                f, scan_var, raw, anode, t, **normalize
            )

            # Append the loaded spectra and steps
            self._spectra.extend(spectra)
            self._steps.extend(steps)

            # Add the measurement number as the id
            m_id = os.path.basename(f)[:3]
            self._m_id.extend([m_id] * len(steps))

        # Convert to numpy arrays
        self._steps = np.asrray(self._steps, dtype=np.float64)
        self._spectra = np.asarray(self._spectra, dtype=object)
        self._m_id = np.asarray(self._m_id, dtype="<U3")

        # Sort the spectra by step values
        inds = np.argsort(self._steps)
        self._steps = self._steps[inds]
        self._spectra = self._spectra[inds]
        self._id = self._id[inds]

        # Initialize attributes
        self.roi = roi  # Region of interest for the detector

    def _load_spectra(
        self,
        file_path: str,
        scan_var: str | None,
        raw: str,
        anode: PositionAnode,
        time_per_step: int | None,
        **normalize: str,
    ) -> tuple[list[Spectrum], list[float]]:
        with h5py.File(file_path, "r") as h5:
            # Load the steps
            if scan_var is not None:
                scan_var += "/0"

                # Check if the step location is valid
                if scan_var not in h5:
                    errmsg = f"{scan_var} not found."
                    raise KeyError(errmsg)

                steps = h5[scan_var]

            # Load the raw data
            group_raw = raw + "/0"

            # Check if the data location is valid
            if group_raw not in h5:
                errmsg = f"{group_raw} not found."

            group_raw = h5[group_raw]

            # Load normalization values
            for name_norm, group_norm in normalize.items():
                group_norm = group_norm + "/0"

                if group_norm not in h5:
                    errmsg = f"{group_norm} not found."
                    raise KeyError(errmsg)

                # Load the dataset / group
                data_norm = h5[group_norm]

                if isinstance(data_norm, h5py.Dataset):
                    # Values averaged by metro
                    normalize[name_norm] = np.asarray(
                        data_norm, dtype=np.float64
                    )

                else:
                    normalize[name_norm] = data_norm

            # Format the data and steps
            spectra = []
            step_val = []

            for i, step in enumerate(group_raw.keys()):
                # Format the normalization values
                step_norm = {}

                # Format the normalization values
                try:
                    for name_norm, data_norm in normalize.items():
                        if isinstance(data_norm, h5.Group):
                            # Check if the step is present
                            if step not in data_norm:
                                raise KeyError()

                            # Calculate the mean and std
                            values = np.asarray(
                                data_norm[step], dtype=np.float64
                            )
                            mean = np.mean(values)
                            std = np.std(values, ddof=1, mean=mean)
                            step_norm[name_norm] = np.asarray(
                                [mean, std]
                            ).flatten()

                        else:
                            # Load the metro averaged value
                            step_norm[name_norm] = np.asarray(
                                [data_norm[i], 0]
                            )

                except Exception:
                    print(f"Skipping {step} due to missing normalization.")
                    continue

                # Format the step value
                step_val.append(float(step))

                if scan_var is not None:
                    try:
                        # Try to read the recorded step value
                        val = float(steps[step][0][0])
                        step_val[i] = val

                    except Exception:
                        print("Step value not found; falling back to index.")

                # Format the raw data
                data = np.asarray(group_raw[step], dtype=np.float64)

                # Initialize the spectrum instance
                spec = Spectrum(
                    anode.process(data), time=time_per_step, **step_norm
                )

                # Append the spectrum
                spectra.append(spec)

        # Return the spectra and energies
        return spectra, step_val

    @property
    def spectra(self) -> NDArray:
        return self._spectra

    @spectra.setter
    def spectra(self, spectra: NDArray) -> None:
        raise AttributeError("Attribute 'spectra' is read-only.")

    @property
    def steps(self) -> NDArray:
        return self._steps

    @steps.setter
    def steps(self, steps: NDArray) -> None:
        if isinstance(steps, np.ndarray):
            if steps.shape != self._spectra.shape:
                errmsg = "steps must be have the same length as spectra."
                raise ValueError(errmsg)

            self._steps = steps

        else:
            raise TypeError("steps must be a numpy array.")

    @property
    def m_id(self) -> NDArray:
        return self._m_id

    @id.setter
    def id(self, measurement_id: NDArray) -> None:
        raise AttributeError("Attribute 'id' is read-only.")

    @property
    def roi(self) -> NDArray:
        return self._roi

    @roi.setter
    def roi(self, roi: ArrayLike) -> None:
        self._roi = parse_roi(roi)

    def select_step_range(
        self,
        step_range: tuple[float, float] | None,
    ) -> tuple[NDArray, NDArray]:
        if step_range is None:
            return self.spectra, self.steps

        step_range = np.asarray(step_range, dtype=np.float64)

        if step_range.shape != (2,):
            errmsg = "step_range must be a tuple (min, max)."
            raise ValueError(errmsg)

        if step_range[1] < step_range[0]:
            errmsg = "step_range max < min."
            raise ValueError(errmsg)

        inds = np.argwhere(
            (self.steps >= step_range[0]) & (self.steps <= step_range[1])
        ).flatten()

        if len(inds) == 0:
            errmsg = f"No steps found in range {step_range}."
            raise RuntimeError(errmsg)

        return self.spectra[inds], self.steps[inds]

    def norm(
        self,
        norm: str,
        step_range: tuple[float, float] | None = None,
    ) -> NDArray:
        """Get the normalization values (and their standard deviation)
        for the specified parameter.

        Parameters
        ----------
        norm: str
            Name of the normalization parameter (`getattr(spectrum, norm)`).
        step_range: [float, float], optional
            Range of step values to consider. If `None`, all steps are
            used.

        Returns
        -------
        val: NDArray, shape (N,)
            The averaged normalization values.
        std: NDArray, shape (N,)
            The respective standard deviations.
        steps: NDArray, shape(N,)
            The corresponding step values.

        """
        # Select the steps and corresponding spectra
        spectra, steps = self.select_step_range(step_range)

        if not hasattr(spectra[0], norm):
            errmsg = f"Unknown normalization {norm}."
            raise AttributeError(errmsg)

        val = np.zeros_like(steps)
        std = np.zeros_like(steps)

        for i in range(len(val)):
            val[i] = getattr(spectra[i], norm)[0]
            std[i] = getattr(spectra[i], norm)[1]

        return val, std, steps

    def transform_norm(self, norm: str, func: callable) -> None:
        """Transform the specified normalization values using a given
        function for all spectra.

        Parameters
        ----------
        norm: str
            Name of the normalization parameter to transform.
        func: callable
            Function to apply to the normalization values. The function
            should take a single argument of type float and return a
            float.

        """
        for spec in self.spectra:
            spec.transform_norm(norm, func)

    def convert_unit(self, norm: str, fro: str, to: str) -> None:
        """Convert the specified normalization values to a different
        unit using the pint package.

        Parameters
        ----------
        norm: str
            Name of the normalization parameter to convert.
        fro: str
            Unit to convert from.
        to: str
            Unit to convert to.

        """
        try:
            from pint import UnitRegistry

        except ImportError as e:
            errmsg = "pint is required to convert units"
            raise ImportError(errmsg) from e

        ureg = UnitRegistry()

        def trafo(x):
            return ureg.Quantity(x, fro).m_as(to)

        self.transform_norm(norm, trafo)

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
        inds = np.argwhere(self.m_id == measurement_id).flatten()

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
        self._id = np.delete(self.id, inds)

    def save(self, filepath: str) -> None:
        """
        Save a scan with pickle.

        Parameters
        ----------
        filepath: str
            Path to the file where the scan will be saved.

        """
        if not filepath.endswith(".pkl") or not filepath.endswith(".pickle"):
            errmsg = "filepath must end with '.pkl' or '.pickle'."
            raise ValueError(errmsg)

        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str) -> BaseScan:
        """
        Load a scan with pickle.

        Parameters
        ----------
        filepath: str
            Path to the file where the scan is saved.

        """
        with open(filepath, "rb") as f:
            scan = pickle.load(f)

        if not isinstance(scan, BaseScan):
            errmsg = "Loaded object is not a scan."
            raise TypeError(errmsg)

        return scan


class Scan(BaseScan):
    """Scan over some variable with a spectrum for each step.

    Parameters
    ----------
    data_files: array_like
        List of data files (str) to be processed.
    anode: PositionAnode
        Anode object to process the raw data.
    scan_var: str, optional
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
        scan_var: str | None = None,
        raw: str = "dld_rd#raw",
        time_per_step: int | ArrayLike | None = None,
        roi: ArrayLike = ((0, 1), (0, 1)),
        qeff: tuple[NDArray, NDArray, NDArray] | None = None,
        bkg: Spectrum | None = None,
        calib: ArrayLike = ((0, 0), (1, 0)),
        **normalize: str,
    ) -> None:
        # Load and process data
        super().__init__(
            data_files,
            anode,
            scan_var=scan_var,
            raw=raw,
            time_per_step=time_per_step,
            roi=roi,
            **normalize,
        )

        # Set attributes
        self.qeff = qeff  # Detector efficiencies
        self.bkg = bkg  # Background spectrum (dark counts)
        self.calib = calib  # Wavelength calibration

    @property
    def qeff(self) -> tuple[NDArray, NDArray, NDArray] | None:
        return self._qeff

    @qeff.setter
    def qeff(
        self, qeff: tuple[NDArray, NDArray, NDArray] | Scan | None
    ) -> None:
        if qeff is None:
            self._qeff = None

        elif isinstance(qeff, Scan):
            self._qeff = parse_qeff(*qeff.qeff)

        else:
            self._qeff = parse_qeff(*qeff)

    @property
    def bkg(self) -> Spectrum | None:
        return self._bkg

    @bkg.setter
    def bkg(self, bkg: Spectrum | None) -> None:
        if bkg is None:
            self._bkg = None

        elif isinstance(bkg, Spectrum):
            self._bkg = bkg

        else:
            errmsg = "bkg must be Spectrum or None."
            raise TypeError(errmsg)

    @property
    def calib(self) -> NDArray:
        return self._calib

    @calib.setter
    def calib(self, calib: ArrayLike) -> None:
        self._calib = parse_calib(calib)

    def counts(
        self,
        bkg: bool = True,
        step_range: tuple[float, float] | None = None,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Get the photon counts for each step.

        Parameters
        ----------
        bkg: bool, optional
            Whether to subtract the background spectrum if available.
        step_range: [float, float], optional
            Range of step values to consider. If None, all steps are
            used.

        Returns
        -------
        n: NDArray, shape (N,)
            The (normalized) counts.
        err: NDArray, shape (N,)
            The respective uncertainties.
        steps: NDArray, shape (N,)
            The corresponding step values.

        """
        # Parse the given calculation options
        bkg = self.bkg if bkg else None

        # Select the steps and corresponding spectra
        spectra, steps = self.select_step_range(step_range)

        n = np.zeros_like(steps)
        err = np.zeros_like(steps)

        for i in range(len(n)):
            n[i], err[i] = spectra[i].counts(roi=self.roi, bkg=bkg)

        return n, err, steps

    def spectrum_at(
        self,
        idx: int,
        bins: int | ArrayLike,
        qeff: bool = True,
        bkg: bool = True,
        calib: bool = True,
        uncertainties: Literal["montecarlo", "poisson"] = "montecarlo",
        mc_samples: int = 10000,
        mc_seed: int | None = None,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Get the spectrum at a specific step.

        Parameters
        ----------
        idx: int
            Index of the step to get the spectrum for.
        bins: int or array_like
            Bin number or edges for the histogram. For a calibrated
            spectrum, these edges should be in wavelength units.
            For an uncalibrated spectrum, these should be between
            0 and 1.
        qeff: bool, optional
            Whether to apply the spatial detector efficiencies if
            available.
        bkg: bool, optional
            Whether to subtract the background spectrum if available.
        calib: bool, optional
            Whether to apply the wavelength calibration if available.
        uncertainties: Literal["montecarlo", "poisson"], optional
            Error propagation method for handling the uncertainties of
            the efficiencies and the wavelength calibration.
        mc_samples: int, optional
            Number of Monte Carlo samples to use for error propagation.
        mc_seed: int, optional
            Seed for the NumPy random generator.

        Returns
        -------
        spec: np.array, shape (N,)
            The (normalized) bin counts.
        err: np.array, shape (N,)
            The uncertainties of the bin counts propagated depending
            on the `uncertainties` argument.
        edges: np.array, shape (N+1,)
            The bin edges.

        """
        # Parse the given calculation options
        qeff = self.qeff if qeff else None
        bkg = self.bkg if bkg else None
        calib = self.calib if calib else ((0, 0), (1, 0))

        # Check index
        if idx >= len(self.steps) or idx < 0:
            errmsg = f"Index {idx} out of range."
            raise IndexError(errmsg)

        # Calculate the spectrum at the specified index
        return self.spectra[idx].spectrum(
            bins,
            roi=self.roi,
            qeff=qeff,
            bkg=bkg,
            calib=calib,
            uncertainties=uncertainties,
            mc_samples=mc_samples,
            mc_seed=mc_seed,
        )

    def spectrum_at_step(
        self,
        step: int | float | str,
        bins: int | ArrayLike,
        qeff: bool = True,
        bkg: bool = True,
        calib: bool = True,
        uncertainties: Literal["montecarlo", "poisson"] = "montecarlo",
        mc_samples: int = 10000,
        mc_seed: int | None = None,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Get the spectrum at a specific step.

        Parameters
        ----------
        step: int or float or str
            Step value to get the spectrum for. The value is
            converted to float and the closest step value
            is used.
        bins: int or array_like
            Bin number or edges for the histogram. For a calibrated
            spectrum, these edges should be in wavelength units.
            For an uncalibrated spectrum, these should be between
            0 and 1.
        qeff: bool, optional
            Whether to apply the spatial detector efficiencies if
            available.
        bkg: bool, optional
            Whether to subtract the background spectrum if available.
        calib: bool, optional
            Whether to apply the wavelength calibration if available.
        uncertainties: Literal["montecarlo", "poisson"], optional
            Error propagation method for handling the uncertainties of
            the efficiencies and the wavelength calibration.
        mc_samples: int, optional
            Number of Monte Carlo samples to use for error propagation.
        mc_seed: int, optional
            Seed for the NumPy random generator.

        Returns
        -------
        spec: np.array, shape (N,)
            The (normalized) bin counts.
        err: np.array, shape (N,)
            The uncertainties of the bin counts propagated depending
            on the `uncertainties` argument.
        edges: np.array, shape (N+1,)
            The bin edges.

        """
        # Find the index of the step closest to the given value
        idx = np.argmin(np.abs(self.steps - float(step)))

        return self.spectrum_at(
            idx,
            bins,
            qeff=qeff,
            bkg=bkg,
            calib=calib,
            uncertainties=uncertainties,
            mc_samples=mc_samples,
            mc_seed=mc_seed,
        )

    def show_spectra(self):
        """Plot the spectra in an interactive window.

        PySide6 or PyQt6 needs to be installed for this to work.

        """
        from agepy.interactive import run
        from ._interactive_scan import SpectrumViewer

        # Intialize the viewer
        mw = SpectrumViewer(self)

        # Run the application
        run(mw)
