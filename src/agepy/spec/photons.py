"""Processing and analysis of fluorescence spectra.

"""
from __future__ import annotations
from typing import Union, Tuple, Sequence
from typing import TYPE_CHECKING
import warnings

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import h5py

from ._anodes import *
from agepy.interactive.photons import AGEScanViewer, QEffViewer
from agepy.interactive import AGEpp

# Import modules for type hinting
if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from iminuit import Minuit



class Spectrum:
    """Fluorescence spectrum.

    """

    def __init__(self,
        xy: np.ndarray,
        time: int = None,
        **norm,
    ) -> None:
        # Store the passed data
        self._xy = xy
        self._t = time
        self._norm = list(norm.keys())
        for key, value in norm.items():
            setattr(self, key, value)

    @classmethod
    def from_h5(cls,
        file_path: str,
        raw: str = "dld_rd#raw/0/0.0",
        time: int = None,
        anode: PositionAnode = DldAnodeUVW(0),
        target_density: str = None,
        intensity_upstream: str = None,
        **norm,
    ) -> None:
        with h5py.File(file_path, "r") as h5:
            # Load the raw data
            raw = np.asarray(h5[raw])
            # Add deprecated normalization parameters for backwards compatibility
            if target_density is not None:
                norm["target_density"] = target_density
            if intensity_upstream is not None:
                norm["intensity_upstream"] = intensity_upstream
            # Load normalization values
            for key, h5path in norm.items():
                if h5path not in h5:
                    raise ValueError("Normalization parameter not found.")
                norm[key] = h5[h5path][0]
        # Initialize the Spectrum
        return cls(anode.process(raw), time=time, **norm)

    def counts(self,
        roi: dict = None,
        qeff: Tuple[np.ndarray, np.ndarray] = None,
        background: Spectrum = None,
    ) -> Tuple[float, float]:
        """Get number of counts in the spectrum and the estimated
        uncertainty.

        Parameters
        ----------
        anode: PositionAnode
            Anode object from `agepy.spec.photons`.
        roi: dict
            Region of interest for the detector. If not provided, the
            full detector is used: `{"x": {"min": 0, "max": 1},
            "y": {"min": 0, "max": 1}}`.
        """
        det_image = np.copy(self._xy)
        # Use the full detector if roi not provided
        if roi is not None:
            # Apply y roi filter
            det_image = det_image[det_image[:,1] > roi["y"]["min"]]
            det_image = det_image[det_image[:,1] < roi["y"]["max"]]
            # Apply x roi filter
            det_image = det_image[det_image[:,0] > roi["x"]["min"]]
            det_image = det_image[det_image[:,0] < roi["x"]["max"]]
        # Apply spatial detector efficiency correction
        if qeff is not None:
            eff, xe = qeff
            x_inds = np.digitize(det_image[:,0], xe)
            # Get the inverse of the efficiency
            inv_eff = 1 / eff[x_inds]
        else:
            inv_eff = np.ones(det_image.shape[0])
        # Calculate the number of counts
        n = np.sum(inv_eff)
        err = np.sqrt(len(inv_eff)) * n / len(inv_eff)
        # Normalize data to measurement duration
        if self._t is not None:
            n /= self._t
            err /= self._t
        # Subtract background before further normalization
        if background is not None:
            bkg_counts, bkg_err = background.counts(roi=roi, qeff=qeff)
            n -= bkg_counts
            n = max(n, 0)
            err = np.sqrt(err**2 + bkg_err**2)
        # Normalize data to account for beam intensity, gas
        # pressure, etc.
        for value in self._norm:
            n /= getattr(self, value)
            err /= getattr(self, value)
        return n, err

    def spectrum(self,
        edges: np.ndarray,
        roi: dict = None,
        qeff: Tuple[np.ndarray, np.ndarray] = None,
        background: Spectrum = None,
        calib: Tuple[float, float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Notes
        -----
        - Implement handling of the uncertainties of the efficiency map
        - Background subtraction is very primitive and uncertainties are
          not propagated correctly
        """
        det_image = np.copy(self._xy)
        # Define the roi as the full detector if not provided
        if roi is None:
            roi = {"x": {"min": 0, "max": 1}, "y": {"min": 0, "max": 1}}
        # Apply y roi filter
        det_image = det_image[det_image[:,1] > roi["y"]["min"]]
        det_image = det_image[det_image[:,1] < roi["y"]["max"]]
        # Don't need y values anymore
        det_image = det_image[:,0].flatten()
        # Apply spatial detector efficiency correction
        if qeff is not None:
            eff, xe = qeff
            x_inds = np.digitize(det_image, xe)
            eff = 1 / eff[x_inds]
        else:
            eff = np.ones(det_image.shape[0])
        # Convert x values to wavelengths
        if calib is not None:
            a, b = calib
            det_image = a * det_image + b
            # Adjust x roi filter to wavelength binning
            wl_min = edges[edges > (a * roi["x"]["min"] + b)][0]
            wl_max = edges[edges < (a * roi["x"]["max"] + b)][-1]
            xroi = (det_image >= wl_min) & (det_image <= wl_max)
        else:
            # Adjust x roi filter to detector binning
            x_min = edges[edges > roi["x"]["min"]][0]
            x_max = edges[edges < roi["x"]["max"]][-1]
            xroi = (det_image >= x_min) & (det_image <= x_max)
        # Apply x roi filter
        det_image = det_image[xroi]
        eff = eff[xroi]
        # Calculate the sum of weights for each bin, i.e. the weighted spectrum
        spectrum = np.histogram(det_image, bins=edges, weights=eff)[0]
        # Histogram data without the per event efficiencies
        hist = np.histogram(det_image, bins=edges)[0]
        # Calculate the uncertainties
        nonzero = hist > 0
        per_bin_eff = np.ones_like(hist, dtype=float)
        per_bin_eff[nonzero] = spectrum[nonzero] / hist[nonzero]
        errors = np.sqrt(hist) * per_bin_eff
        # Normalize data to measurement duration per step
        if self._t is not None:
            spectrum /= self._t
            errors /= self._t
        # Subtract background before further normalization
        if background is not None:
            bkg_spec, bkg_err = background.spectrum(
                edges, roi=roi, qeff=qeff, calib=calib
            )
            spectrum -= bkg_spec
            spectrum[spectrum < 0] = 0
            errors = np.sqrt(errors**2 + bkg_err**2)
        # Normalize data to account for beam intensity, gas 
        # pressure, etc.
        for value in self._norm:
            spectrum /= getattr(self, value)
            errors /= getattr(self, value)
        # Return the spectrum and uncertainties
        return spectrum, errors

    def transform_norm(self, norm: str, func: callable):
        setattr(self, norm, func(getattr(self, norm)))

    def convert_unit(self, norm: str, fro: str, to: str):
        try:
            from pint import UnitRegistry

        except ImportError:
            raise ImportError("pint is required to convert units.")
        ureg = UnitRegistry()
        # Convert the normalization values
        val = getattr(self, norm)
        setattr(self, norm, ureg.Quantity(val, fro).m_as(to))


class Scan:
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
    target_density: str, optional
        Path to the target density in the data files. Default: None.
    intensity_downstream: str, optional
        Path to the downstream intensity in the data files. Default:
        None.
    intensity_upstream: str, optional
        Path to the upstream intensity in the data files. Default:
        None.

    Attributes
    ----------
    anode: PositionAnode
        Anode object from `agepy.spec.photons`.
    spectra: np.ndarray
        Array of the loaded Spectrum objects.
    steps: np.ndarray
        Array of the scan variable values.

    Notes
    -----
    - Very minimal implementation, needs to be expanded

    """

    def __init__(self,
        data_files: Sequence[str],
        anode: PositionAnode,
        scan_var: str = None,
        raw: str = "dld_rd#raw/0",
        time_per_step: int = None,
        roi: dict = None,
        target_density: str = None,
        intensity_upstream: str = None,
        **norm,
    ) -> None:
        self.roi = roi
        self.spectra = []
        self.steps = []
        if isinstance(data_files, str):
            data_files = [data_files]
        # Add deprecated normalization parameters for backwards compatibility
        if target_density is not None:
            norm["target_density"] = target_density
        if intensity_upstream is not None:
                norm["intensity_upstream"] = intensity_upstream
        for f in data_files:
            spec, steps = self._load_spectra(
                f, scan_var, raw, anode, time_per_step, **norm)
            self.spectra.extend(spec)
            self.steps.extend(steps)
        # Convert to numpy arrays
        self.steps = np.array(self.steps)
        self.spectra = np.array(self.spectra)
        # Sort the spectra by step values
        _sort = np.argsort(self.steps)
        self.steps = self.steps[_sort]
        self.spectra = self.spectra[_sort]

    def _load_spectra(self,
        file_path: str,
        scan_var: str,
        raw: str,
        anode: PositionAnode,
        time_per_step: int,
        **norm,
    ) -> Tuple[list, list]:
        with h5py.File(file_path, "r") as h5:
            # Load the steps
            if scan_var is not None:
                steps = h5[scan_var]
            # Load the raw data
            raw = h5[raw]
            # Load normalization values
            for key, h5path in norm.items():
                norm[key] = np.asarray(h5[h5path])
            # Format the data and steps
            spectra = []
            step_val = []
            for i, step in enumerate(raw.keys()):
                # Format the step value
                if scan_var is not None:
                    step_val.append(steps[step][0][0])
                else:
                    step_val.append(float(step))
                # Format the raw data
                data = np.asarray(raw[step])
                # Format the normalization values
                step_norm = {key: value[i] for key, value in norm.items()}
                # Initialize the spectrum instance
                spectra.append(Spectrum(anode.process(data),
                                        time=time_per_step, **step_norm))
        # Return the spectra and energies
        return spectra, step_val

    def save(self, filepath: str) -> None:
        """
        Save a scan with pickle.

        """
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str) -> Scan:
        """
        Load a scan with pickle.

        """
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def counts(self,
        roi: dict = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the photon-excitation energy spectrum.

        Parameters
        ----------
        roi: dict
            Region of interest for the detector. If not provided, the
            full detector is used.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            The number of counts (normalized), the respective
            statistical uncertainties, and the exciting-photon energies.

        """
        if roi is None:
            roi = self.roi
        vectorized_counts = np.vectorize(
            lambda spec: spec.counts(roi=roi)
        )
        n, err = vectorized_counts(self.spectra)
        return n, err, self.steps

    def spectrum_at(self,
        step: Union[int, float],
        edges: np.ndarray,
        roi: dict = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the spectrum at a specific step.

        """
        if step not in self.steps:
            raise ValueError("Step value not found in scan.")
        if roi is None:
            roi = self.roi
        spec = self.spectra[self.steps == step]
        if len(spec) > 1:
            warnings.warn("Multiple spectra found for step value. "
                          "Returning the first.")
        spec = spec[0]
        return spec.spectrum(edges, roi=roi)

    def convert_unit(self, norm: str, fro: str, to: str):
        try:
            from pint import UnitRegistry

        except ImportError:
            raise ImportError("pint is required to convert units.")
        ureg = UnitRegistry()
        trafo = lambda x: ureg.Quantity(x, fro).m_as(to)
        for spec in self.spectra:
            spec.transform_norm(norm, trafo)

    def show_spectra(self):
        """Plot the spectra in an interactive window.

        """
        app = AGEpp(AGEScanViewer, self)
        app.run()


class EnergyScan(Scan):
    """Scan over exciting-photon energies with a spectrum for each step.

    Parameters
    ----------
    data_files: Sequence[str]
        List of data files to be processed.
    anode: PositionAnode
        Anode object from `agepy.spec.photons`.
    energies: str, optional
        Path to the energy values in the data files. If None,
        the keys are used as the values.
    raw: str, optional
        Path to the raw data in the data files. Default:
        "dld_rd#raw/0".
    time_per_step: int, optional
        Time per step in the scan. Default: None.
    target_density: str, optional
        Path to the target density in the data files. Default: None.
    intensity_downstream: str, optional
        Path to the downstream intensity in the data files. Default:
        None.
    intensity_upstream: str, optional
        Path to the upstream intensity in the data files. Default:
        None.

    Attributes
    ----------
    spectra: np.ndarray
        Array of the loaded Spectrum objects.
    energies: np.ndarray
        Array of the scan variable values.

    Notes
    -----
    - Very minimal implementation, needs to be expanded

    """

    def __init__(self,
        data_files: Sequence[str],
        anode: PositionAnode,
        energies: str = None,
        raw: str = "dld_rd#raw/0",
        time_per_step: int = None,
        roi: dict = None,
        target_density: str = None,
        intensity_upstream: str = None,
        **norm,
    ) -> None:
        super().__init__(data_files, anode, energies, raw, time_per_step, roi,
                         target_density, intensity_upstream, **norm)

    @property
    def energies(self) -> np.ndarray:
        return self.steps

    @energies.setter
    def energies(self, value: np.ndarray) -> None:
        self.steps = value


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

    def __init__(self,
        data_files: Sequence[str],
        anode: PositionAnode,
        raw: str = "dld_rd#raw/0",
        time_per_step: int = None,
        roi: dict = None,
        binning: int = 24,
        model: str = "gaussian",
        **norm,
    ) -> None:
        super().__init__(data_files, anode, None, raw, time_per_step, roi, **norm)
        n = len(self.steps)
        if binning > n:
            warnings.warn("Binning larger than number of recorded steps. "
                          "Setting binning to number of steps.")
            binning = n
        # Set the model for the fits
        self.model = model
        # Initialize the result arrays
        _, self.xe = np.histogram([], bins=binning, range=(0, 1))
        self._py = np.full(n, np.nan, dtype=np.float64)
        self._pyerr = np.full(n, np.nan, dtype=np.float64)
        self._px = np.full(n, np.nan, dtype=np.float64)

    def efficiencies(self) -> Tuple[np.ndarray, np.ndarray]:
        # Calculate the efficiencies for each bin
        _efficiencies = np.zeros(len(self.xe)-1, dtype=np.float64)
        _errors = np.zeros(len(self.xe)-1, dtype=np.float64)
        inds = np.digitize(self._px, self.xe)
        for i in range(len(_efficiencies)):
            c = 0
            for j in np.argwhere(inds == i).flatten():
                if self._py[j] == np.nan:
                    continue
                _efficiencies[i] += self._py[j]
                _errors[i] += self._pyerr[j]**2
                c += 1
            if c >= 1:
                _efficiencies[i] /= c
                _errors[i] = np.sqrt(_errors[i]) / c
        # Interpolate the efficiencies, if necessary
        for i in range(1, len(_efficiencies)-1):
            if _efficiencies[i] == 0:
                if (_efficiencies[i-1] != 0) and (_efficiencies[i+1] != 0):
                    _efficiencies[i] = (_efficiencies[i-1] + _efficiencies[i+1]) * 0.5
                    _errors[i] = (_errors[i-1] + _errors[i+1]) * 0.5
        # Normalize the efficiencies to maximum value
        m = _efficiencies.max()
        return _efficiencies / m, _errors / m

    def fit_highest_peak(self) -> None:
        edges = np.histogram([], bins=1024, range=(0, 1))[1]
        fit_range = 25
        for i in range(len(self.steps)):
            # Prepare data
            spec, err = self.spectra[i].spectrum(edges, roi=self.roi)
            peak = np.argmax(spec)
            spec = spec[peak-fit_range:peak+fit_range]
            err = err[peak-fit_range:peak+fit_range]
            xe = edges[peak-fit_range:peak+fit_range+1]
            n = np.stack((spec, err**2), axis=-1)
            # Peform the fit
            self._fit_peak(i, n, xe)

    def _fit_in_range(self, i: int, fit_range: Tuple[float, float]) -> None:
        edges = np.histogram([], bins=1024, range=(0, 1))[1]
        # Prepare data
        spec, err = self.spectra[i].spectrum(edges, roi=self.roi)
        peak = np.argwhere((edges > fit_range[0]) & (edges < fit_range[1])).flatten()
        spec = spec[peak[:-1]]
        err = err[peak[:-1]]
        xe = edges[peak]
        n = np.stack((spec, err**2), axis=-1)
        # Peform the fit
        return self._fit_peak(i, n, xe)


    def _fit_peak(self, i: int, n: np.ndarray, xe: np.ndarray) -> None:
        try:
            from iminuit import Minuit
            from iminuit.cost import ExtendedBinnedNLL
            from numba_stats import norm, voigt

        except ImportError:
            raise ImportError("iminuit and numba-stats is required for fitting.")

        # Prepare the fit
        if self.model == "gaussian":
            def model(x, s, loc, scale):
                return s * norm.cdf(x, loc, scale)

            start = (n.max(), (xe[-1] + xe[0]) * 0.5, 0.01)
            limits = {"s": (0, None), "loc": (xe[0], xe[-1]), "scale": (0.0001, 0.1)}
            use_pdf = ''
        elif self.model == "voigt":
            def model(x, s, gamma, loc, scale):
                return s * voigt.pdf(x, gamma, loc, scale)

            start = (n.max(), 0.01, (xe[-1] + xe[0]) * 0.5, 0.01)
            limits = {"s": (0, None), "gamma": (0.0001, 0.1), "loc": (xe[0], xe[-1]), "scale": (0.0001, 0.1)}
            use_pdf = "numerical"
        else:
            raise ValueError("Invalid model.")
        cost = ExtendedBinnedNLL(n, xe, model, use_pdf=use_pdf)
        m = Minuit(cost, *start)
        for par in limits:
            m.limits[par] = limits[par]
        # Perform the fit
        m.migrad()
        # Store the results
        if not m.valid:
            self._py[i] = np.nan
            self._pyerr[i] = np.nan
            self._px[i] = np.nan
        else:
            self._py[i] = m.values["s"]
            self._pyerr[i] = m.errors["s"]
            self._px[i] = m.values["loc"]
        return m

    def _add_fit_result(self, i: int, y: float, err: float, x: float) -> None:
        self._py[i] = y
        self._pyerr[i] = err
        self._px[i] = x

    def interactive(self):
        """Plot the spectra in an interactive window.

        """
        app = AGEpp(QEffViewer, self)
        app.run()

    def plot_eff(self, ax: Axes = None, color: str = "k", label: str = None) -> Tuple[Figure, Axes]:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        eff, err = self.efficiencies()
        nonzero = np.argwhere(eff > 0).flatten()
        eff = eff[nonzero[0]:nonzero[-1]+1]
        err = err[nonzero[0]:nonzero[-1]+1]
        dx = (self.xe[1] - self.xe[0]) * 0.5
        x = self.xe[nonzero[0]:nonzero[-1]+1] + dx
        ax.errorbar(x, eff, yerr=err, xerr=dx, fmt="+", color=color, label=label)
        ax.set_xlabel("Detector Position [arb. u.]")
        ax.set_ylabel("Efficiency [arb. u.]")
        ax.set_xlim(0, 1)
        ax.set_title("Detector Efficiencies")
        return fig, ax
