"""Processing and analysis of fluorescence spectra."""

from __future__ import annotations

import pickle
import numpy as np
import h5py

from .spectrum import Spectrum


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Union, Tuple, Literal, Sequence
    from numpy.typing import ArrayLike, NDArray
    from agepy.spec.photons.anodes import PositionAnode

    type RegionOfInterest = Tuple[Tuple[float, float], Tuple[float, float]]
    type QuantumEfficiency = Tuple[NDArray, NDArray, NDArray]
    type WavelengthCalib = Tuple[Tuple[float, float], Tuple[float, float]]
    type ErrorPropagation = Literal["montecarlo", "none"]


class BaseScan:
    def __init__(
        self,
        data_files: Sequence[str],
        anode: PositionAnode,
        scan_var: str | None = None,
        raw: str = "dld_rd#raw",
        time_per_step: int | Sequence[int] | None = None,
        roi: RegionOfInterest | None = None,
        **norm: str,
    ) -> None:
        # Intialize lists for the spectra and steps
        self._spectra = []
        self._steps = []

        # Keep track of the measurement number
        self._id = []

        # Force data_files to be a list
        if isinstance(data_files, str):
            data_files = [data_files]

        # Create a list of time_per_step values if only one is provided
        if isinstance(time_per_step, int):
            time_per_step = [time_per_step] * len(data_files)

        # Check if the length of time_per_step matches the number of data files
        if len(time_per_step) != len(data_files):
            raise ValueError(
                "time_per_step must be a single int or a list of the same "
                "length as data_files."
            )

        # Load the spectra from the data files
        for f, t in zip(data_files, time_per_step):
            spec, steps = self._load_spectra(
                f, scan_var, raw, anode, t, **norm
            )
            self._spectra.extend(spec)
            self._steps.extend(steps)

            # Add the measurement number as the id
            f = os.path.basename(f)[:3]
            self._id.extend([f] * len(steps))

        # Convert to numpy arrays
        self._steps = np.array(self._steps)
        self._spectra = np.array(self._spectra)
        self._id = np.array(self._id)

        # Sort the spectra by step values
        _sort = np.argsort(self._steps)
        self._steps = self._steps[_sort]
        self._spectra = self._spectra[_sort]
        self._id = self._id[_sort]

        # Initialize attributes
        self.roi = roi  # Region of interest for the detector

    def _load_spectra(
        self,
        file_path: str,
        scan_var: str,
        raw: str,
        anode: PositionAnode,
        time_per_step: int,
        **norm: str,
    ) -> Tuple[list[Spectrum], list[float]]:
        with h5py.File(file_path, "r") as h5:
            # Load the steps
            if scan_var is not None:
                steps = h5[scan_var + "/0"]

            # Load the raw data
            raw = h5[raw + "/0"]

            # Load normalization values
            for key, h5path in norm.items():
                if h5path.endswith("avg"):
                    norm[key] = np.asarray(h5[h5path + "/0"])

                else:
                    norm[key] = h5[h5path + "/0"]

            # Format the data and steps
            spectra = []
            step_val = []

            for i, step in enumerate(raw.keys()):
                # Format the normalization values
                step_norm = {}

                try:
                    for key, value in norm.items():
                        if isinstance(value, np.ndarray):
                            step_norm[key] = value[i]

                        else:
                            values = np.asarray(value[step])
                            step_norm[key] = np.array(
                                [np.mean(values), np.std(values)]
                            )

                except Exception:
                    print("Skipping step due to missing normalization values.")
                    continue

                # Format the step value
                if scan_var is not None:
                    try:
                        step_val.append(steps[step][0][0])

                    except Exception:
                        print("Step value not found. Falling back to index.")
                        step_val.append(float(step))

                else:
                    step_val.append(float(step))

                # Format the raw data
                data = np.asarray(raw[step])

                # Initialize the spectrum instance
                spectra.append(
                    Spectrum(
                        anode.process(data), time=time_per_step, **step_norm
                    )
                )

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
            if steps.shape == (len(self.spectra),):
                self._steps = steps

            else:
                raise ValueError(
                    "steps must be a 1D array of the same length as spectra."
                )
        else:
            raise ValueError("steps must be a numpy array.")

    @property
    def id(self) -> NDArray:
        return self._id

    @id.setter
    def id(self, measurement_id: NDArray) -> None:
        raise AttributeError("Attribute 'id' is read-only.")

    @property
    def roi(self) -> RegionOfInterest:
        return self._roi

    @roi.setter
    def roi(self, roi: RegionOfInterest) -> None:
        self._roi = self.prepare_roi(roi)

    def prepare_roi(
        self,
        roi: RegionOfInterest | bool | None,
    ) -> RegionOfInterest:
        if roi is None:
            return None

        if isinstance(roi, bool):
            if roi:
                return self.roi

            else:
                return None

        else:
            try:
                roi = np.array(roi, dtype=np.float64)
                assert roi.shape == (2, 2)

            except Exception as e:
                errmsg = "roi must be provided as ((xmin, xmax), (ymin, ymax))"
                raise ValueError(errmsg) from e

            return roi

    def select_step_range(
        self,
        step_range: Tuple[float, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if step_range is not None:
            try:
                mask = (self.steps >= step_range[0]) & (
                    self.steps <= step_range[1]
                )
                steps = self.steps[mask]
                spectra = self.spectra[mask]

            except Exception as e:
                errmsg = "step_range must be (step_min, step_max)"
                raise ValueError(errmsg) from e

            return spectra, steps

        else:
            return self.spectra, self.steps

    def norm(
        self,
        norm: str,
        step_range: Tuple[float, float] = None,
    ) -> np.ndarray:
        """Get the normalization values (and their standard deviation)
        for the specified parameter.

        Parameters
        ----------
        norm: str
            Name of the normalization parameter (`getattr(spectrum, norm)`).
        step_range: Tuple[float, float], optional
            Range of step values to consider. If None, all steps are
            used.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            The averaged normalization values, the respective standard
            deviation, and the corresponding step values.

        """
        # Select the steps and corresponding spectra
        spectra, steps = self.select_step_range(step_range)

        val = np.zeros_like(steps)
        std = np.zeros_like(steps)

        for i in range(len(val)):
            if isinstance(getattr(spectra[i], norm), np.ndarray):
                val[i] = getattr(spectra[i], norm)[0]
                std[i] = getattr(spectra[i], norm)[1]

            else:
                val[i] = getattr(spectra[i], norm)

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
        measurement_number: str,
        steps: Union[Sequence[int], Sequence[float]],
    ) -> np.ndarray:
        """Remove the specified steps of a measurement from the scan.

        Parameters
        ----------
        measurement_number: str
            Measurement number (metro) to remove the steps from.
        steps: Union[Sequence[int], Sequence[float]]
            List of step values to remove.

        """
        # Select the spectra corresponding to the measurement number
        mask = np.argwhere(self.id == measurement_number).flatten()

        # Check if the measurement number exists
        if len(mask) == 0:
            raise ValueError("Measurement number not found.")

        # Select the steps to remove
        inds = []
        for step in steps:
            # Select the index of the closest step value
            inds.append(np.argsort(np.abs(self.steps[mask] - step))[0])

        # Check if the steps exist
        if len(inds) == 0:
            raise ValueError("Step values not found.")

        mask = mask[inds]

        # Remove the steps
        self._steps = np.delete(self.steps, mask)
        self._spectra = np.delete(self.spectra, mask)
        self._id = np.delete(self.id, mask)

        return mask

    def save(self, filepath: str) -> None:
        """
        Save a scan with pickle.

        Parameters
        ----------
        filepath: str
            Path to the file where the scan will be saved.

        """
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str) -> Scan:
        """
        Load a scan with pickle.

        Parameters
        ----------
        filepath: str
            Path to the file where the scan is saved.

        """
        with open(filepath, "rb") as f:
            return pickle.load(f)


class Scan(BaseScan):
    """Scan over some variable with a spectrum for each step.

    Parameters
    ----------
    data_files: Sequence[str]
        List of data files to be processed.
    anode: PositionAnode
        Anode object from `agepy.spec.photons`.
    scan_var: str, optional
        Path to the step values in the data files. If None,
        the keys are used as the values.
    raw: str, optional
        Path to the raw data in the data files. Default:
        "dld_rd#raw/0".
    time_per_step: int, optional
        Time per step in the scan. Default: None.
    roi: Tuple[Tuple[float, float], Tuple[float, float]], optional
        Region of interest for the detector in the form
        `((xmin, xmax), (ymin, ymax))`. Default: None.
    **norm
        Normalization parameters as keyword arguments with the name as
        key and the h5 path as the value. For example,
        `intensity_upstream = mirror#value`.

    Attributes
    ----------
    anode: PositionAnode
        Anode object from `agepy.spec.photons`.
    spectra: np.ndarray
        Array of the loaded Spectrum objects.
    steps: np.ndarray
        Array of the scan variable values.
    id: np.ndarray
        Array of the measurement numbers.
    roi: Tuple[Tuple[float, float], Tuple[float, float]]
        Region of interest for the detector.
    qeff: Tuple[np.ndarray, np.ndarray, np.ndarray]
        Detector efficiencies in the form `(values, errors, x)`.
    bkg: Spectrum
        Background spectrum (dark counts) to be subtracted.
    calib: Tuple[Tuple[float, float], Tuple[float, float]]
        Wavelength calibration parameters in the form
        `((a0, err), (a1, err))`.

    """

    def __init__(
        self,
        data_files: Sequence[str],
        anode: PositionAnode,
        scan_var: str = None,
        raw: str = "dld_rd#raw",
        time_per_step: Union[int, Sequence[int]] = None,
        roi: Tuple[Tuple[float, float], Tuple[float, float]] = None,
        **norm,
    ) -> None:
        # Load and process data
        super().__init__(
            data_files, anode, scan_var, raw, time_per_step, roi, **norm
        )

        # Set attributes
        self._qeff = None  # Detector efficiencies
        self._bkg = None  # Background spectrum (dark counts)
        self._calib = None  # Wavelength calibration

    @property
    def qeff(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._qeff

    @qeff.setter
    def qeff(self, qeff: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        self._qeff = self.prepare_qeff(qeff)

    @property
    def bkg(self) -> Spectrum:
        return self._bkg

    @bkg.setter
    def bkg(self, bkg: Spectrum) -> None:
        self._bkg = self.prepare_bkg(bkg)

    @property
    def calib(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return self._calib

    @calib.setter
    def calib(
        self, calib: Tuple[Tuple[float, float], Tuple[float, float]]
    ) -> None:
        self._calib = self.prepare_calib(calib)

    def prepare_qeff(
        self,
        qeff: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        if qeff is None:
            return None

        if isinstance(qeff, bool):
            if qeff:
                return self.qeff

            else:
                return None

        if isinstance(qeff, QEffScan):
            return qeff.qeff

        try:
            values = np.array(qeff[0])
            errors = np.array(qeff[1])
            x = np.array(qeff[2])

            # Check if the arrays have the same length
            assert len(values) == len(errors)
            assert len(values) == len(x)
            # Check if the values and errors are positive
            assert np.all(values > 0)
            assert np.all(errors >= 0)
            # Check if x is between 0 and 1
            assert np.all(x > 0)
            assert np.all(x < 1)

        except Exception as e:
            errmsg = "qeff could not be parsed"
            raise ValueError(errmsg) from e

        return values, errors, x

    def prepare_bkg(
        self,
        bkg: Spectrum,
    ) -> Spectrum:
        if bkg is None:
            return None

        if isinstance(bkg, bool):
            if bkg:
                return self.bkg

            else:
                return None

        elif isinstance(bkg, Spectrum):
            return bkg

        else:
            raise ValueError("bkg could not be parsed.")

    def prepare_calib(
        self,
        calib: Tuple[Tuple[float, float], Tuple[float, float]],
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        if calib is None:
            return None

        if isinstance(calib, bool):
            if calib:
                return self.calib

            else:
                return None

        try:
            calib_params = np.array(calib, dtype=np.float64)
            assert calib_params.shape == (2, 2)

        except Exception as e:
            errmsg = "calib could not be parsed"
            raise ValueError(errmsg) from e

        return calib_params

    def counts(
        self,
        roi: Tuple[Tuple[float, float], Tuple[float, float]] = None,
        qeff: bool = True,
        bkg: bool = True,
        step_range: Tuple[float, float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the photon-excitation energy spectrum.

        Parameters
        ----------
        roi: Tuple[Tuple[float, float], Tuple[float, float]], optional
            Region of interest for the detector in the form
            `((xmin, xmax), (ymin, ymax))`. Defaults to the roi set for
            the `Scan` instance.
        qeff: bool, optional
            Whether to apply the spatial detector efficiencies if
            available.
        bkg: bool, optional
            Whether to subtract the background spectrum if available.
        step_range: Tuple[float, float], optional
            Range of step values to consider. If None, all steps are
            used.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            The number of counts (normalized), the respective
            uncertainties, and the corresponding step values.

        """
        # Parse the given calculation options
        roi = self.prepare_roi(roi)
        qeff = self.prepare_qeff(qeff)
        bkg = self.prepare_bkg(bkg)

        # Select the steps and corresponding spectra
        spectra, steps = self.select_step_range(step_range)

        n = np.zeros_like(steps)
        err = np.zeros_like(steps)

        for i in range(len(n)):
            n[i], err[i] = spectra[i].counts(roi=roi, qeff=qeff, bkg=bkg)

        return n, err, steps

    def spectrum_at(
        self,
        idx: int,
        edges: np.ndarray,
        roi: Tuple[Tuple[float, float], Tuple[float, float]] = None,
        qeff: bool = True,
        bkg: bool = True,
        calib: bool = True,
        err_prop: Literal["montecarlo", "none"] = "montecarlo",
        mc_samples: int = 10000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the spectrum at a specific step.

        Parameters
        ----------
        idx: int
            Index of the step to get the spectrum for.
        edges: np.ndarray
            Bin edges for the histogram. For a calibrated spectrum,
            these should be in wavelength units. For an uncalibrated
            spectrum, these should be between 0 and 1.
        roi: Tuple[Tuple[float, float], Tuple[float, float]], optional
            Region of interest for the detector in the form
            `((xmin, xmax), (ymin, ymax))`. If not provided, the
            full detector is used.
        qeff: bool, optional
            Whether to apply the spatial detector efficiencies if
            available.
        bkg: bool, optional
            Whether to subtract the background spectrum if available.
        calib: bool, optional
            Whether to apply the wavelength calibration if available.
        err_prop: Literal["montecarlo", "none"], optional
            Error propagation method for handling the uncertainties of
            the efficiencies and the wavelength calibration. If
            `qeff = None` and `calib = None` and `bkg = None`, this
            setting has no effect. Can be 'montecarlo', or 'none'.
        mc_samples: int, optional
            Number of Monte Carlo samples to use for error propagation.
            Has no effect if `err_prop = 'none'`.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The spectrum and its uncertainties.

        """
        # Parse the given calculation options
        roi = self.prepare_roi(roi)
        qeff = self.prepare_qeff(qeff)
        bkg = self.prepare_bkg(bkg)
        calib = self.prepare_calib(calib)

        # Calculate the spectrum at the specified index
        return self.spectra[idx].spectrum(
            edges,
            roi=roi,
            qeff=qeff,
            bkg=bkg,
            calib=calib,
            err_prop=err_prop,
            mc_samples=mc_samples,
        )

    def spectrum_at_step(
        self,
        step: Union[int, float],
        edges: np.ndarray,
        roi: Tuple[Tuple[float, float], Tuple[float, float]] = None,
        qeff: bool = True,
        bkg: bool = True,
        calib: bool = True,
        err_prop: Literal["montecarlo", "none"] = "montecarlo",
        mc_samples: int = 10000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the spectrum at a specific step.

        Parameters
        ----------
        step: Union[int, float]
            Step value to get the spectrum for. The closest step value
            is used.
        edges: np.ndarray
            Bin edges for the histogram. For a calibrated spectrum,
            these should be in wavelength units. For an uncalibrated
            spectrum, these should be between 0 and 1.
        roi: Tuple[Tuple[float, float], Tuple[float, float]], optional
            Region of interest for the detector in the form
            `((xmin, xmax), (ymin, ymax))`. If not provided, the
            full detector is used.
        qeff: bool, optional
            Whether to apply the spatial detector efficiencies if
            available.
        bkg: bool, optional
            Whether to subtract the background spectrum if available.
        calib: bool, optional
            Whether to apply the wavelength calibration if available.
        err_prop: Literal["montecarlo", "none"], optional
            Error propagation method for handling the uncertainties of
            the efficiencies and the wavelength calibration. If
            `qeff = None` and `calib = None` and `bkg = None`, this
            setting has no effect. Can be 'montecarlo', or 'none'.
        mc_samples: int, optional
            Number of Monte Carlo samples to use for error propagation.
            Has no effect if `err_prop = 'none'`.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The spectrum and its uncertainties.

        """
        # Find the index of the step closest to the given value
        idx = np.argmin(np.abs(self.steps - step))

        return self.spectrum_at(
            idx,
            edges,
            roi=roi,
            qeff=qeff,
            bkg=bkg,
            calib=calib,
            err_prop=err_prop,
            mc_samples=mc_samples,
        )

    def show_spectra(self):
        """Plot the spectra in an interactive window."""
        from agepy.interactive import run
        from agepy.spec.interactive.photons_scan import SpectrumViewer

        # Intialize the viewer
        mw = SpectrumViewer(self)

        # Run the application
        run(mw)
