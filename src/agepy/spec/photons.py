"""Processing and analysis of fluorescence spectra.

"""
from __future__ import annotations
from typing import Union, Tuple, Sequence
import warnings

import numpy as np
import h5py

from ._anodes import *
from agepy.interactive.photons import AGEScanViewer
from agepy.interactive import AGEpp


class Spectrum:
    """Fluorescence spectrum.

    """

    def __init__(self,
        raw: np.ndarray,
        time: int = None,
        anode: PositionAnode = None,
        **norm,
    ) -> None:
        # Store the passed data
        self._raw = raw
        self._t = time
        self.anode = anode
        self._norm = list(norm.keys())
        for key, value in norm.items():
            setattr(self, key, value)

    @classmethod
    def from_h5(cls,
        file_path: str,
        raw: str = "dld_rd#raw/0/0.0",
        time: int = None,
        anode: PositionAnode = None,
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
        return cls(raw, time=time, anode=anode, **norm)

    def counts(self,
        anode: PositionAnode = None,
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
        # Get the xy values
        if anode is None:
            anode = self.anode
        if anode is None:
            raise ValueError("Anode object must be provided for processing.")
        det_image = anode.process(self._raw)
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
            bkg_counts, bkg_err = background.counts(anode, roi=roi, qeff=qeff)
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
        anode: PositionAnode = None,
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
        # Get the xy values
        if anode is None:
            anode = self.anode
        if anode is None:
            raise ValueError("Anode object must be provided for processing.")
        det_image = anode.process(self._raw)
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
                edges, anode, roi=roi, qeff=qeff, calib=calib
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
        target_density: str = None,
        intensity_upstream: str = None,
        **norm,
    ) -> None:
        self.anode = anode  
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
                f, scan_var=scan_var, raw=raw, time_per_step=time_per_step,
                anode=None, **norm
            )
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
        scan_var: str = None,
        raw: str = "dld_rd#raw/0",
        time_per_step: int = None,
        anode: PositionAnode = None,
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
                spectra.append(Spectrum(data, time=time_per_step, anode=anode,
                                        **step_norm))
        # Return the spectra and energies
        return spectra, step_val

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
        vectorized_counts = np.vectorize(
            lambda spec: spec.counts(self.anode, roi=roi)
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
        spec = self.spectra[self.steps == step]
        if len(spec) > 1:
            warnings.warn("Multiple spectra found for step value. "
                          "Returning the first.")
        spec = spec[0]
        return spec.spectrum(edges, anode=self.anode, roi=roi)

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
    anode: PositionAnode
        Anode object from `agepy.spec.photons`.
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
        target_density: str = None,
        intensity_upstream: str = None,
        **norm,
    ) -> None:
        super().__init__(data_files, anode, energies, raw, time_per_step,
                         target_density, intensity_upstream, **norm)

    @property
    def energies(self) -> np.ndarray:
        return self.steps

    @energies.setter
    def energies(self, value: np.ndarray) -> None:
        self.steps = value
