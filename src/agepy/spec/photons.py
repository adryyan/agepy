"""Processing and analysis of fluorescence spectra.

"""
from __future__ import annotations
from typing import Union, Tuple, Sequence, Dict
from typing import TYPE_CHECKING
import warnings

import os
import pickle
import numpy as np
from numba import njit, prange
from scipy.interpolate import CubicSpline
import pandas as pd
import matplotlib.pyplot as plt
import h5py

from ._anodes import *
from agepy.interactive.photons import AGEScanViewer, QEffViewer, PhexViewer, PhemViewer
from agepy.interactive.util import import_jacobi_propagate, import_iminuit
propagate = import_jacobi_propagate()
from agepy.interactive import AGEpp

# Import modules for type hinting
if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from iminuit import Minuit



@njit()
def compute_bin(x, bin_edges):
    # assuming uniform bins
    n = bin_edges.shape[0] - 1
    a_min = bin_edges[0]
    a_max = bin_edges[-1]

    # special case to mirror NumPy behavior for last bin
    if x == a_max:
        return n - 1 # a_max always in last bin

    bin = int(n * (x - a_min) / (a_max - a_min))

    if bin < 0 or bin >= n:
        return None
    else:
        return bin


@njit()
def numba_histogram(data, bin_edges, weights):
    hist = np.zeros((bin_edges.shape[0] - 1,), dtype=np.intp)

    for x, w in zip(data.flat, weights.flat):
        bin = compute_bin(x, bin_edges)
        if bin is not None:
            hist[int(bin)] += w

    return hist


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
        raw: str = "dld_rd#raw",
        time: int = None,
        anode: PositionAnode = DldAnodeUVW(0),
        target_density: str = None,
        intensity_upstream: str = None,
        **norm,
    ) -> None:
        with h5py.File(file_path, "r") as h5:
            # Load the raw data
            raw = np.asarray(h5[raw + "/0/0.0"])
            # Add deprecated normalization parameters for backwards compatibility
            if target_density is not None:
                norm["target_density"] = target_density
            if intensity_upstream is not None:
                norm["intensity_upstream"] = intensity_upstream
            # Load normalization values
            for key, h5path in norm.items():
                if h5path not in h5:
                    raise ValueError("Normalization parameter not found.")
                norm[key] = h5[h5path + "/0"][0]
        # Initialize the Spectrum
        return cls(anode.process(raw), time=time, **norm)

    def counts(self,
        roi: dict = None,
        qeff: Tuple[np.ndarray, np.ndarray, np.ndarray] = None,
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
            eff, eff_err, xe = qeff
            x_inds = np.digitize(det_image[:,0], xe[1:-1])
            # Get the inverse of the efficiency
            eff = eff[x_inds]
            eff_err = eff_err[x_inds]
            nonzero = eff > 0
            eff = eff[nonzero]
            eff_err = eff_err[nonzero]
            inv_eff = 1 / eff
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
            # Using just the statistical uncertainty of the background
            # counts would underestimate the uncertainty of the subtraction
            bkg_err = np.sqrt(bkg_counts * self._t) / self._t
            n -= bkg_counts
            n = max(n, 0)
            err = np.sqrt(err**2 + bkg_err**2)
        # Normalize data to account for beam intensity, gas
        # pressure, etc.
        for norm in self._norm:
            if isinstance(getattr(self, norm), np.ndarray):
                val, err_val = getattr(self, norm)
                err = np.sqrt(err**2 / val**2 + err_val**2 * n**2 / val**4)
                n /= val
            else:
                n /= getattr(self, norm)
                err /= getattr(self, norm)
        return n, err

    def spectrum(self,
        edges: np.ndarray,
        roi: dict = None,
        qeff: Tuple[np.ndarray, np.ndarray, np.ndarray] = None,
        background: Spectrum = None,
        calib: Tuple[float, float] = None,
        uncertainties: str = "jacobi",
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
            eff, eff_err, eff_xe = qeff
            x_inds = np.digitize(det_image, eff_xe[1:-1])
        # Prepare the calibration parameters
        if calib is not None:
            a0 = calib[0]
            a1 = calib[1]
        else:
            a0 = (0, 0)
            a1 = (1, 0)
        # Adjust x roi filter to wavelength binning
        wl_min = edges[edges > (a1[0] * roi["x"]["min"] + a0[0])][0]
        wl_max = edges[edges < (a1[0] * roi["x"]["max"] + a0[0])][-1]
        # Histogram data without the per event efficiencies
        wl = a1[0] * det_image + a0[0]
        xroi = (wl >= wl_min) & (wl <= wl_max)
        wl = wl[xroi]
        hist = np.histogram(wl, bins=edges)[0]
        if calib is None and qeff is None:
            spectrum = hist
            errors = np.sqrt(spectrum)
        elif uncertainties == "jacobi":
            if qeff is None:

                def sum_weights(a):
                    # Convert x values to wavelengths
                    wl = a[1] * det_image + a[0]
                    xroi = (wl >= wl_min) & (wl <= wl_max)
                    wl = wl[xroi]
                    return np.histogram(wl, bins=edges)[0]

                spectrum, errors = propagate(sum_weights, (a0[0], a1[0]), (a0[1]**2, a1[1]**2))
                errors = np.sqrt(np.diag(errors))
            else:

                def sum_weights(eff, a0, a1):
                    # Get the efficiencies for each point
                    peff = eff[x_inds].flatten()
                    nonzero = peff > 0
                    peff[nonzero] = 1 / peff[nonzero]
                    # Convert x values to wavelengths
                    wl = a1 * det_image + a0
                    xroi = (wl >= wl_min) & (wl <= wl_max)
                    wl = wl[xroi]
                    peff = peff[xroi]
                    return np.histogram(wl, bins=edges, weights=peff)[0]

                if calib is None:
                    spectrum, errors = propagate(lambda x: sum_weights(x, a0[0], a1[0]), eff, eff_err**2)
                    errors = np.sqrt(np.diag(errors))
                else:
                    x = np.append(eff, [a0[0], a1[0]])
                    xcov = np.append(eff_err**2, [a0[1]**2, a1[1]**2])
                    spectrum, errors = propagate(lambda x: sum_weights(x[:-2], x[-2], x[-1]), x, xcov)
                    errors = np.sqrt(np.diag(errors))

        elif uncertainties == "MonteCarlo":
            n = 10000
            rng = np.random.default_rng()
            # Create n samples of the calibration parameters
            if calib is None:
                a0_samples = np.zeros(n)
                a1_samples = np.ones(n)
            else:
                a0_samples = rng.normal(loc=a0[0], scale=a0[1], size=n)
                a1_samples = rng.normal(loc=a1[0], scale=a1[1], size=n)
            # Create n samples of the efficiencies
            if qeff is None:
                eff_samples = np.ones((n, len(det_image)))
                x_inds = np.arange(len(det_image))
            else:
                eff_samples = rng.normal(loc=eff, scale=eff_err, size=(n, len(eff)))
            # Initialize array for storing the sample results
            spectrum = np.zeros((n, len(edges) - 1))

            @njit(parallel=True)
            def sum_weights(spectrum):
                for i in prange(n):
                    # Get the efficiencies for each point
                    eff = eff_samples[i][x_inds].flatten()
                    nonzero = eff > 0
                    eff[nonzero] = 1 / eff[nonzero]
                    # Convert x values to wavelengths
                    wl = a1_samples[i] * det_image + a0_samples[i]
                    # Adjust x roi filter to wavelength binning
                    xroi = (wl >= wl_min) & (wl <= wl_max)
                    # Apply x roi filter
                    wl = wl[xroi]
                    eff = eff[xroi]
                    # Calculate the sum of weights for each bin, i.e. the weighted spectrum
                    spectrum[i] = numba_histogram(wl, edges, eff)
                return spectrum

            # Calculate the weighted spectrum for each sample
            spectrum = sum_weights(spectrum)
            # Calculate mean and standard deviation of the sampled spectra
            errors = np.std(spectrum, axis=0)
            spectrum = np.mean(spectrum, axis=0)

        # Include the Poisson uncertainties
        if calib is not None or qeff is not None:
            nonzero = hist > 0
            per_bin_eff = np.ones_like(hist, dtype=float)
            per_bin_eff[nonzero] = spectrum[nonzero] / hist[nonzero]
            errors = np.sqrt(np.sqrt(hist)**2 * per_bin_eff**2 + errors**2)

        # Normalize data to measurement duration per step
        if self._t is not None:
            spectrum /= self._t
            errors /= self._t

        # Subtract background before further normalization
        if background is not None:
            bkg_spec, bkg_err = background.spectrum(
                edges, roi=roi, qeff=qeff, calib=calib, uncertainties="jacobi"
            )
            # Using just the statistical uncertainty of the background
            # counts would underestimate the uncertainty of the subtraction
            bkg_err = np.sqrt(bkg_spec * self._t) / self._t
            spectrum -= bkg_spec
            spectrum[spectrum < 0] = 0
            errors = np.sqrt(errors**2 + bkg_err**2)

        # Normalize data to account for beam intensity, gas 
        # pressure, etc.
        for norm in self._norm:
            if isinstance(getattr(self, norm), np.ndarray):
                val, err_val = getattr(self, norm)
                errors = np.sqrt(errors**2 / val**2 + err_val**2 * spectrum**2 / val**4)
                spectrum /= val
            else:
                spectrum /= getattr(self, norm)
                errors /= getattr(self, norm)

        # Return the spectrum and uncertainties
        return spectrum, errors

    def transform_norm(self, norm: str, func: callable):
        val = getattr(self, norm)
        if isinstance(val, np.ndarray):
            val, err = propagate(func, val[0], val[1]**2)
            setattr(self, norm, np.array([val, np.sqrt(err)]))
        else:
            setattr(self, norm, func(val))

    def convert_unit(self, norm: str, fro: str, to: str):
        try:
            from pint import UnitRegistry

        except ImportError:
            raise ImportError("pint is required to convert units.")
        ureg = UnitRegistry()
        # Convert the normalization values
        func = lambda x: ureg.Quantity(x, fro).m_as(to)
        self.transform_norm(norm, func)


class BaseScan:

    def __init__(self,
        data_files: Sequence[str],
        anode: PositionAnode,
        scan_var: str = None,
        raw: str = "dld_rd#raw",
        time_per_step: Union[int, Sequence[int]] = None,
        roi: dict = None,
        target_density: str = None,
        intensity_upstream: str = None,
        **norm,
    ) -> None:
        self.spectra = []
        self.steps = []
        self.id = []
        if isinstance(data_files, str):
            data_files = [data_files]
        if isinstance(time_per_step, int):
            time_per_step = [time_per_step] * len(data_files)
        if len(time_per_step) != len(data_files):
            raise ValueError("time_per_step must be a single int or a list "
                             "of the same length as data_files.")
        # Add deprecated normalization parameters for backwards compatibility
        if target_density is not None:
            norm["target_density"] = target_density
        if intensity_upstream is not None:
                norm["intensity_upstream"] = intensity_upstream
        for f, t in zip(data_files, time_per_step):
            spec, steps = self._load_spectra(f, scan_var, raw, anode, t, **norm)
            self.spectra.extend(spec)
            self.steps.extend(steps)
            # Add the measurement number as the id
            f = os.path.basename(f)[:3]
            self.id.extend([f] * len(steps))
        # Convert to numpy arrays
        self.steps = np.array(self.steps)
        self.spectra = np.array(self.spectra)
        self.id = np.array(self.id)
        # Sort the spectra by step values
        _sort = np.argsort(self.steps)
        self.steps = self.steps[_sort]
        self.spectra = self.spectra[_sort]
        self.id = self.id[_sort]
        # Initialize attributes
        self.roi = roi  # Region of interest for the detector
        self.qeff = None  # Detector efficiencies
        self.bkg = None  # Background spectrum (dark counts)
        self.calib = None  # Wavelength calibration

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
                # Format the step value
                if scan_var is not None:
                    step_val.append(steps[step][0][0])
                else:
                    step_val.append(float(step))
                # Format the raw data
                data = np.asarray(raw[step])
                # Format the normalization values
                step_norm = {}
                for key, value in norm.items():
                    if isinstance(value, np.ndarray):
                        step_norm[key] = value[i]
                    else:
                        values = np.asarray(value[step])
                        step_norm[key] = np.array([np.mean(values), np.std(values)])
                # Initialize the spectrum instance
                spectra.append(Spectrum(anode.process(data),
                                        time=time_per_step, **step_norm))
        # Return the spectra and energies
        return spectra, step_val

    def counts(self,
        roi: dict = None,
        qeff: bool = True,
        xrange: Tuple[float, float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the photon-excitation energy spectrum.

        Parameters
        ----------
        roi: dict, optional
            Ignore set region of interest and use the provided one instead. 

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            The number of counts (normalized), the respective
            statistical uncertainties, and the exciting-photon energies.

        """
        if roi is None:
            roi = self.roi
        if self.qeff is None or not qeff:
            qeff = None
        else:
            edges = np.histogram([], bins=512, range=(0, 1))[1]
            qeff = (*self.qeff.efficiencies(edges), edges)

        vectorized_counts = np.vectorize(
            lambda spec: spec.counts(roi=roi, qeff=qeff, background=self.bkg)
        )
        if xrange is not None:
            mask = (self.steps >= xrange[0]) & (self.steps <= xrange[1])
            n, err = vectorized_counts(self.spectra[mask])
            return n, err, self.steps[mask]
        else:
            n, err = vectorized_counts(self.spectra)
            return n, err, self.steps

    def norm(self, norm: str) -> np.ndarray:
        return np.array([getattr(spec, norm) for spec in self.spectra])

    def transform_norm(self, norm: str, func: callable):
        for spec in self.spectra:
            spec.transform_norm(norm, func)

    def convert_unit(self, norm: str, fro: str, to: str):
        try:
            from pint import UnitRegistry

        except ImportError:
            raise ImportError("pint is required to convert units.")
        ureg = UnitRegistry()
        trafo = lambda x: ureg.Quantity(x, fro).m_as(to)
        self.transform_norm(norm, trafo)

    def remove_steps(self,
        measurement_number: str,
        steps: Union[Sequence[int], Sequence[float]],
    ) -> None:
        mask = np.argwhere(self.id == measurement_number).flatten()
        inds = []
        for step in steps:
            inds.append(np.argsort(np.abs(self.steps[mask] - step))[0])
        mask = mask[inds]
        # Remove the steps
        self.steps = np.delete(self.steps, mask)
        self.spectra = np.delete(self.spectra, mask)
        self.id = np.delete(self.id, mask)

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

    def set_qeff(self, qeff: QEffScan) -> None:
        self.qeff = qeff

    def set_bkg(self, bkg: Spectrum) -> None:
        self.bkg = bkg

    def set_calib(self, a0: float, a1: float) -> None:
        self.calib = (a0, a1)

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
        raw: str = "dld_rd#raw",
        time_per_step: Union[int, Sequence[int]] = None,
        roi: dict = None,
        target_density: str = None,
        intensity_upstream: str = None,
        **norm,
    ) -> None:
        super().__init__(data_files, anode, energies, raw, time_per_step, roi,
                         target_density, intensity_upstream, **norm)
        self.phex_assignments = None
        self.phex_label = None
        self.phem_assignments = None
        self.phem_label = None

    @property
    def energies(self) -> np.ndarray:
        return self.steps

    @energies.setter
    def energies(self, value: np.ndarray) -> None:
        self.steps = value

    def assign_phex(self,
        reference: pd.DataFrame,
        label: Dict[str, Union[Sequence[str], int]],
        energy_range: float,
        simulation: pd.DataFrame = None,
    ) -> None:
        """Calibrate the exciting-photon energies.

        """
        self.phex_label = label
        app = AGEpp(PhexViewer, self, reference, label, energy_range, simulation)
        app.run()

    def plot_phex(self,
        reference: pd.DataFrame,
        plot_pulls: bool = True,
    ) -> Minuit:
        fit_energies = []
        ref_energies = []
        labels = []
        for i, row in self.phex_assignments.iterrows():
            ref = reference.copy()
            label = ""
            for key, value in row.items():
                if key in ["E", "val", "err"]:
                    continue
                ref.query(f"{key} == @value", inplace=True)
                label += f"{key} = {value}, "
            label = label[:-2]
            if ref.empty:
                continue
            fit_energies.append([row["val"][1], row["err"][1]])
            ref_energies.append(ref["E"].iloc[0])
            labels.append(label)
        fit_energies = np.array(fit_energies)
        ref_energies = np.array(ref_energies)
        labels = np.array(labels)
        # Sort by calibration energies
        idx = np.argsort(ref_energies)
        ref_energies = ref_energies[idx]
        fit_energies = fit_energies[idx]
        labels = labels[idx]
        # Define the Model
        def model(x, b0, b1):
            return b1 * x + b0
        # Create the cost function
        Minuit, cost = import_iminuit()
        c = cost.LeastSquares(ref_energies, fit_energies[:,0], fit_energies[:,1], model)
        # Initialize the minimizer
        m = Minuit(c, b1=1, b0=0)
        m.limits["b1"] = (0.9, 1.1)
        m.limits["b0"] = (-0.1, 0.1)
        m.migrad()
        # Check pulls
        pulls = np.abs(c.pulls(m.values))
        inds = np.argwhere(pulls > 5).flatten()
        for idx in inds:
            print("Bad assignment of:", labels[idx], "with diff:", c.prediction(m.values)[idx] - fit_energies[idx,0])
        # Plot the results
        fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={"hspace": 0.3})
        ax[0].errorbar(ref_energies, fit_energies[:,0], yerr=fit_energies[:,1],
                       fmt="s", markersize=1.5, label="Assign. Phex")
        ax[0].plot(ref_energies, c.prediction(m.values), label="Lin. Regr.")
        #chi2ndof = m.fmin.reduced_chi2
        #ax[0].legend(title=r"$\chi^2\;/\;$ndof = " + f"{chi2ndof:.2f}", loc="best")
        ax[0].legend()
        ax[0].set_title(r"Assigned Photon-Excitation Energies $E_\text{data}$")
        ax[0].set_ylabel(r"$E_\text{data}$ [eV]")
        ax[1].axhline(0, color="black", linestyle="--", alpha=0.9)
        if plot_pulls:
            ax[1].step(ref_energies, c.pulls(m.values), where="mid")
            ax[1].set_title("Studentized Residuals of the Linear Regression")
            ax[1].set_ylabel("Pulls")
        else:
            ax[1].step(ref_energies, c.prediction(m.values) - fit_energies[:,0], where="mid")
            ax[1].set_title("Difference to the Linear Regression")
            ax[1].set_ylabel(r"$(\text{Lin. Regr.} - E_\text{data})$ [eV]")
        ax[1].set_xlabel(r"$E_\text{literature}$ [eV]")
        return fig, ax

    def save_phex(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.phex_assignments, f)

    def load_phex(self, path: str) -> None:
        with open(path, "rb") as f:
            self.phex_assignments = pickle.load(f)

    def assign_phem(self,
        reference: pd.DataFrame,
        phem_label: Dict[str, Union[Sequence[str], int]],
        phex_label: Sequence[str],
        calib_guess: Tuple[float, float],
    ) -> None:
        """Calibrate the exciting-photon energies.

        """
        app = AGEpp(PhemViewer, self, reference, phem_label, phex_label, calib_guess)
        app.run()

    def plot_phem(self,
        reference: pd.DataFrame,
        plot_pulls: bool = True,
        start: Tuple[float, float] = (100, 40),
    ) -> Minuit:
        fit_wl = []
        ref_wl = []
        labels = []
        for i, row in self.phem_assignments.iterrows():
            ref = reference.copy()
            label = ""
            for key, value in row.items():
                if key in ["val", "err"]:
                    continue
                if key not in ref.columns:
                    continue
                ref.query(f"{key} == @value", inplace=True)
                label += f"{key} = {value}, "
            label = label[:-2]
            if ref.empty:
                continue
            fit_wl.append([row["val"][1], row["err"][1]])
            ref_wl.append(ref["E"].iloc[0])
            labels.append(label)
        fit_wl = np.array(fit_wl)
        ref_wl = np.array(ref_wl)
        labels = np.array(labels)
        # Sort by calibration energies
        idx = np.argsort(ref_wl)
        ref_wl = ref_wl[idx]
        fit_wl = fit_wl[idx]
        labels = labels[idx]
        # Define the Model
        def model(x, b0, b1):
            return b1 * x + b0
        # Create the cost function
        Minuit, cost = import_iminuit()
        c = cost.LeastSquares(ref_wl, fit_wl[:,0], fit_wl[:,1], model)
        # Parse start values
        b1start = 1 / start[1]
        b0start = -start[0] / start[1]
        # Initialize the minimizer
        m = Minuit(c, b1=b1start, b0=b0start)
        m.migrad()
        # Set calibration
        b0, b1 = m.values["b0"], m.values["b1"]
        b0err, b1err = m.errors["b0"], m.errors["b1"]
        a1 = 1 / b1
        a1err = b1err / b1**2
        a0 = -b0 / b1
        a0err = np.sqrt(b0err**2 / b1**2 + b0**2 * b1err**2 / b1**4)
        self.calib = ((a0, a0err), (a1, a1err))
        # Check pulls
        pulls = np.abs(c.pulls(m.values))
        inds = np.argwhere(pulls > 5).flatten()
        for idx in inds:
            print("Bad assignment of:", labels[idx], "with diff:", c.prediction(m.values)[idx] - fit_wl[idx,0])
        # Plot the results
        fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={"hspace": 0.3})
        ax[0].errorbar(ref_wl, fit_wl[:,0], yerr=fit_wl[:,1],
                       fmt="s", markersize=1.5, label="Assign. Phem")
        ax[0].plot(ref_wl, c.prediction(m.values), label="Lin. Regr.")
        #chi2ndof = m.fmin.reduced_chi2
        #ax[0].legend(title=r"$\chi^2\;/\;$ndof = " + f"{chi2ndof:.2f}", loc="best")
        ax[0].legend()
        ax[0].set_title(r"Assigned Detector Positions $x_\text{data}$")
        ax[0].set_ylabel(r"$x_\text{data}$ [arb. u.]")
        ax[1].axhline(0, color="black", linestyle="--", alpha=0.9)
        if plot_pulls:
            ax[1].step(ref_wl, c.pulls(m.values), where="mid")
            ax[1].set_title("Studentized Residuals of the Linear Regression")
            ax[1].set_ylabel("Pulls")
        else:
            ax[1].step(ref_wl, c.prediction(m.values) - fit_wl[:,0], where="mid")
            ax[1].set_title("Difference to the Linear Regression")
            ax[1].set_ylabel(r"$(\text{Lin. Regr.} - x_\text{data})$ [arb. u.]")
        ax[1].set_xlabel(r"$\lambda_\text{literature}$ [nm]")
        return fig, ax

    def save_phem(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.phem_assignments, f)

    def load_phem(self, path: str) -> None:
        with open(path, "rb") as f:
            self.phem_assignments = pickle.load(f)

    def assigned_spectrum(self,
        phex: Dict[str, Union[str, int]],
        edges: np.ndarray,
        qeff: bool = True,
        bkg: bool = True,
        calib: bool = True,
        norm_phex: bool = False,
        uncertainties: str = "accurate",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the spectrum with assigned energies.

        """
        df = self.phex_assignments.copy()
        for key, value in phex.items():
            if df.empty:
                raise ValueError("Phex assignment not found.")
            df.query(f"{key} == @value", inplace=True)
        # At this point there should be only one assignment
        assert len(df) == 1, "Multiple or no assignments found."
        fit_val = df["val"].iloc[0]
        fit_err = df["err"].iloc[0]
        # Find clossest energy step
        n = 1.2
        step_idx = np.argwhere(np.abs(self.steps - fit_val[1]) < fit_val[2] * n).flatten()
        if len(step_idx) == 0:
            raise ValueError("No steps found.")
        # Define energy range
        erange = (fit_val[1] - fit_val[2] * n, fit_val[1] + fit_val[2] * n)
        # Check if multiple phex assignments overlap
        overlap = self.phex_assignments.query("E > @erange[0] and E < @erange[1]")
        for i, row in overlap.iterrows():
            if i == df.index[0]:
                continue
            _val = row["val"]
            _step_idx = np.argwhere(np.abs(self.steps - _val[1]) < _val[2] * n).flatten()
            # Remove the overlapping steps
            _step_idx = np.setdiff1d(step_idx, _step_idx)
            if len(_step_idx) == 0:
                warnings.warn("No steps found without overlap.")
            else:
                step_idx = _step_idx
        # Get quantum efficiency
        if qeff and self.qeff is not None:
            xe = np.histogram([], bins=512, range=(0, 1))[1]
            qeff = (*self.qeff.efficiencies(xe), xe)
        else:
            qeff = None
        # Get background spectrum
        if bkg and self.bkg is not None:
            bkg = self.bkg
        else:
            bkg = None
        # Get wavelength calibration
        if calib and self.calib is not None:
            calib = self.calib
        else:
            calib = None
        # Combine the specta
        spec = np.zeros(len(edges) - 1)
        err = np.zeros(len(edges) - 1)
        for idx in step_idx:
            s, e = self.spectra[idx].spectrum(edges, roi=self.roi, qeff=qeff, background=bkg, calib=calib, uncertainties=uncertainties)
            if norm_phex:
                s /= fit_val[0]
                e = np.sqrt(e**2 / fit_val[0]**2 + s**2 * fit_err[0]**2 / fit_val[0]**4)
            spec += s
            err += e**2
        err = np.sqrt(err)
        if norm_phex:
            spec /= len(step_idx)
            err /= len(step_idx)
        return spec, err

    def phexphem(self,
        xedges: np.ndarray = None,
        yedges: np.ndarray = None,
        qeff: bool = True,
        bkg: bool = True,
        calib: bool = True,
    ) -> np.ndarray:
        """
        
        """
        if xedges is None:
            dx = np.mean(np.diff(self.steps) * 0.5)
            dec = -int(np.floor(np.log10(dx)))
            dx = round(dx, dec)
            xmin = round(np.min(self.steps), dec) - dx
            xedges = np.arange(xmin, np.max(self.steps) + dx * 2, dx * 2)
        # Get quantum efficiency
        if qeff and self.qeff is not None:
            xe = np.histogram([], bins=512, range=(0, 1))[1]
            qeff = (*self.qeff.efficiencies(xe), xe)
        else:
            qeff = None
        # Get background spectrum
        if bkg and self.bkg is not None:
            bkg = self.bkg
        else:
            bkg = None
        # Get wavelength calibration
        if calib and self.calib is not None:
            calib = self.calib
            if yedges is None:
                yedges = np.histogram([], bins=512, range=(0, 1))[1]
                yedges = calib[1] * yedges + calib[0]
        else:
            yedges = np.histogram([], bins=512, range=(0, 1))[1]
            calib = None
        # Create an empty map
        hist = np.zeros((len(yedges) - 1, len(xedges) - 1))
        # Fill the map
        weights, _ = np.histogram(self.steps, bins=xedges)
        if len(weights[weights > 1]) > 0:
            warnings.warn("Multiple spectra found for the same energy.")
        inds = np.digitize(self.steps, xedges[1:-1])
        for spec, idx in zip(self.spectra, inds):
            hist[:, idx] += spec.spectrum(
                yedges,
                roi=self.roi,
                qeff=qeff,
                background=bkg,
                calib=calib,
            )[0]
        # Normalize the map
        weights[weights == 0] = 1
        hist /= weights
        return hist, xedges, yedges


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
        raw: str = "dld_rd#raw",
        time_per_step: Union[int, Sequence[int]] = None,
        roi: dict = None,
        model: str = "gaussian",
        **norm,
    ) -> None:
        # Force the x roi to cover the full detector
        if roi is not None:
            roi["x"]["min"] = 0
            roi["x"]["max"] = 1
        super().__init__(data_files, anode, None, raw, time_per_step, roi, **norm)
        # Set the model for the fits
        self.model = model
        # Initialize the result arrays
        n = len(self.steps)
        self._py = np.full(n, np.nan, dtype=np.float64)
        self._pyerr = np.full(n, np.nan, dtype=np.float64)
        self._px = np.full(n, np.nan, dtype=np.float64)

    def efficiencies(self, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Get center of bins
        dx = (edges[1] - edges[0]) * 0.5
        x = edges[:-1] + dx
        # Get the fit values
        py, pyerr, px = self.fit_results()
        # Interpolate the efficiencies
        def interp(y):
            spl = CubicSpline(px, y, bc_type="natural")
            return spl(x)
        # Generate samples
        n = 10000
        rng = np.random.default_rng()
        y_samples = rng.normal(loc=py, scale=pyerr, size=(n, len(py)))
        eff_samples = np.stack([interp(y) for y in y_samples], axis=0)
        # Calculate the mean and standard deviation
        eff = np.mean(eff_samples, axis=0)
        err = np.std(eff_samples, axis=0)
        # Calculate the distance to the closest value in px for each value in x
        #distances = np.abs(x[:, np.newaxis] - px[np.newaxis, :])
        #min_distances = np.min(distances, axis=1)
        # Add a penalty term dependend on the distance to the closest value
        # This is kinda arbitrary and one should think about a better way to
        # penalize interpolated values that are far away from the fit values
        #err += err * min_distances * 100
        # Set values outside the interpolation range to 0
        eff[x < px[0]] = 0
        eff[x > px[-1]] = 0
        err[x < px[0]] = 0
        err[x > px[-1]] = 0
        return eff, err

    def efficiencies_fit(self, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Get center of bins
        dx = (edges[1] - edges[0]) * 0.5
        x = edges[:-1] + dx
        # Get the fit values
        py, pyerr, px = self.fit_results()
        # Fit a third order polynomial to the fit values
        try:
            from iminuit import Minuit
            from iminuit.cost import LeastSquares
            from numba_stats import bernstein
            from jacobi import propagate

        except ImportError:
            raise ImportError("iminuit and numba-stats is required for fitting.")

        # Prepare the fit
        def model(x, *par):
            return bernstein.density(x, par, px[0], px[-1])

        cost = LeastSquares(px, py, pyerr, model)
        m = Minuit(cost, (1, 1, 1, 1, 1), name=("a0", "a1", "a2", "a3", "a4"))
        m.limits["a0", "a1", "a2", "a3", "a4"] = (0, None)
        m.migrad()
        if not m.valid:
            try:
                m.interactive()
            except ImportError:
                raise ImportError("PySide6 is required for fit debugging.")
        # Get the efficiency values and uncertainties
        eff, err = propagate(lambda par: model(x, *par), m.values, m.covariance)
        err = np.sqrt(np.diag(err))
        # Set values outside the interpolation range to 0
        eff[x < px[0]] = 0
        eff[x > px[-1]] = 0
        err[x < px[0]] = 0
        err[x > px[-1]] = 0
        return eff, err

    def fit_results(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Remove NaN values
        mask = ~np.isnan(self._px)
        py = self._py[mask]
        pyerr = self._pyerr[mask]
        px = self._px[mask]
        # Sort the values
        sort = np.argsort(px)
        return py[sort] / py.max(), pyerr[sort] / py.max(), px[sort]

    def fit_highest_peak(self) -> None:
        edges = np.histogram([], bins=512, range=(0, 1))[1]
        fit_range = 20
        for i in range(len(self.steps)):
            # Prepare data
            spec, err = self.spectra[i].spectrum(edges, roi=self.roi)
            peak = np.argmax(spec)
            spec = spec[peak-fit_range:peak+fit_range]
            err = err[peak-fit_range:peak+fit_range]
            xe = edges[peak-fit_range:peak+fit_range+1]
            n = np.stack((spec, err**2), axis=-1)
            # Peform the fit
            m = self._fit_peak(i, n, xe)
            if not m.valid:
                try:
                    m.interactive()
                except ImportError:
                    raise ImportError("PySide6 is required for fit debugging.")
                else:
                    self._add_fit_result(i, m.values["s"], m.errors["s"], m.values["loc"])

    def _fit_in_range(self, i: int, fit_range: Tuple[float, float]) -> None:
        edges = np.histogram([], bins=512, range=(0, 1))[1]
        # Prepare data
        spec, err = self.spectra[i].spectrum(edges, roi=self.roi)
        peak = np.argwhere((edges > fit_range[0]) & (edges < fit_range[1])).flatten()
        spec = spec[peak[:-1]]
        err = err[peak[:-1]]
        xe = edges[peak]
        n = np.stack((spec, err**2), axis=-1)
        # Peform the fit
        return n, xe


    def _fit_peak(self, i: int, n: np.ndarray, xe: np.ndarray) -> None:
        try:
            from iminuit import Minuit
            from iminuit.cost import ExtendedBinnedNLL
            from numba_stats import norm, voigt, crystalball_ex, qgaussian

        except ImportError:
            raise ImportError("iminuit and numba-stats is required for fitting.")

        # Prepare the fit
        if self.model == "Gaussian":
            def model(x, *par):
                return par[0] * norm.cdf(x, *par[1:])

            start = (n.max(), (xe[-1] + xe[0]) * 0.5, 0.01)
            limits = {"s": (0, None), "loc": (xe[0], xe[-1]), "scale": (0.0001, 0.1)}
            use_pdf = ""
        elif self.model == "Voigt":
            def model(x, *par):
                return par[0] * voigt.pdf(x, *par[1:])

            start = (n.max(), 0.01, (xe[-1] + xe[0]) * 0.5, 0.01)
            limits = {"s": (0, None), "gamma": (0.0001, 0.1), "loc": (xe[0], xe[-1]), "scale": (0.0001, 0.1)}
            use_pdf = "numerical"
        elif self.model == "CrystalballEx":
            def model(x, *par):
                return par[0] * crystalball_ex.cdf(x, *par[1:])

            start = (n.max(), 1, 1.5, 0.01, 1, 1.5, 0.01, (xe[-1] + xe[0]) * 0.5)
            limits = {"s": (0, None), "beta_left": (0, 5), "m_left": (1, 20), "scale_left": (0.0001, 0.1), "beta_right": (0, 5), "m_right": (1, 20), "scale_right": (0.0001, 0.1), "loc": (xe[0], xe[-1])}
            use_pdf = ""
        elif self.model == "Q-Gaussian":
            def model(x, *par):
                return par[0] * qgaussian.cdf(x, *par[1:])

            start = (n.max(), 1.5, (xe[-1] + xe[0]) * 0.5, 0.01)
            limits = {"s": (0, None), "q": (1, 3), "loc": (xe[0], xe[-1]), "scale": (0.0001, 0.1)}
            use_pdf = ""
        else:
            raise ValueError("Invalid model.")
        cost = ExtendedBinnedNLL(n, xe, model, use_pdf=use_pdf)
        m = Minuit(cost, *start, name=limits.keys())
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
        # Plot fit values
        y, yerr, x = self.fit_results()
        ax.errorbar(x, y, yerr=yerr, fmt="s", color=color, label=label)
        ylim = ax.get_ylim()
        ax.set_ylim(ylim)
        # Plot the interpolated values
        xe = np.histogram([], bins=1024, range=(0, 1))[1]
        x = xe[:-1] + (xe[1] - xe[0]) * 0.5
        eff, err = self.efficiencies(xe)
        ax.plot(x, eff, color=color, linestyle="-")
        ax.fill_between(x, eff - err, eff + err, color=color, alpha=0.3)
        # Set ylim to auto
        ax.set_ylim(auto=True)
        ax.set_xlabel("Detector Position [arb. u.]")
        ax.set_ylabel("Efficiency [arb. u.]")
        ax.set_xlim(0, 1)
        ax.set_title("Detector Efficiencies")
        return fig, ax

    def set_qeff(self, qeff: QEffScan) -> None:
        raise NotImplementedError("QEffScan cannot set QEffScan.")
