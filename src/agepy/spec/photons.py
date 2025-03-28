"""Processing and analysis of fluorescence spectra.

"""
from __future__ import annotations
import warnings
import os
import pickle

# Import dependencies
import numpy as np
from numba import njit, prange
from jacobi import propagate
from scipy.stats import norm
import pandas as pd
from matplotlib import gridspec
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import h5py

# Import internal modules
from ._anodes import *

# Import modules for type hinting
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Union, Tuple, Sequence, Dict, Literal
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from iminuit import Minuit
    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "QEffScan", "Spectrum", "Scan", "EnergyScan", "available_anodes",
    "PocoAnode", "WsaAnode", "DldAnodeXY", "DldAnodeUVW", "DldAnodeUV",
    "DldAnodeUW", "DldAnodeVW", "Old_WsaAnode", "Old_DldAnode"
]


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
def numba_histogram(data, bin_edges):
    hist = np.zeros((bin_edges.shape[0] - 1,), dtype=np.float64)

    for x in data.flat:
        bin = compute_bin(x, bin_edges)
        if bin is not None:
            hist[int(bin)] += 1

    return hist


@njit()
def numba_weighted_histogram(data, bin_edges, weights):
    hist = np.zeros((bin_edges.shape[0] - 1,), dtype=np.float64)

    for x, w in zip(data.flat, weights.flat):
        bin = compute_bin(x, bin_edges)
        if bin is not None:
            hist[int(bin)] += w

    return hist


@njit(parallel=True)
def _mc_calibrated_spectrum(output, data, edges, calib, rng, n):
    # Determine number of measured data points
    data_counts = len(data)

    # Start the Monte Carlo simulation
    for i in prange(n):
        # Select data points based on Poisson sampling
        p = rng.poisson(lam=data_counts, size=1)[0]
        data_sample = rng.choice(data, size=p)
        # Convert x values to wavelengths
        a0_sample = rng.normal(calib[0][0], calib[0][1], size=1)[0]
        a1_sample = rng.normal(calib[1][0], calib[1][1], size=1)[0]
        data_sample = a1_sample * data_sample + a0_sample
        # Calculate the sum of weights for each bin, i.e. the weighted spectrum
        output[i] = numba_histogram(data_sample, edges)

    # Return the n generated Monte Carlo spectra
    return output


@njit(parallel=True)
def _mc_calibrated_spectrum_with_qeff(output, data, edges, qeff, calib, rng, n):
    # Prepare the quantum efficiency correction
    n_eff = len(qeff[0])
    # Define the interpolation grid for the efficiencies
    xedges = np.linspace(0, 1, 513)
    xvalues = (xedges[1:] + xedges[:-1]) * 0.5
    # Assign the efficiencies to the data points
    eff_inds = np.digitize(data, xedges[1:-1])

    # Determine number of measured data points
    data_counts = len(data)

    # Start the Monte Carlo simulation
    for i in prange(n):
        # Create a sample of the efficiencies
        eff_sample = np.ones(n_eff)
        for j in range(n_eff):
            eff_sample[j] = rng.normal(qeff[0][j], qeff[1][j], size=1)[0]
        # Interpolate the efficiencies to get a smoother spectrum
        eff_sample = np.interp(xvalues, qeff[2], eff_sample)
        # Get the efficiencies for each point
        data_eff = 1 / eff_sample[eff_inds]
        # Select data points based on Poisson sampling
        p = rng.poisson(lam=data_counts, size=1)[0]
        poisson_inds = rng.integers(0, data_counts, size=p)
        data_sample = data[poisson_inds]
        data_eff = data_eff[poisson_inds]
        # Convert x values to wavelengths
        a0_sample = rng.normal(calib[0][0], calib[0][1], size=1)[0]
        a1_sample = rng.normal(calib[1][0], calib[1][1], size=1)[0]
        data_sample = a1_sample * data_sample + a0_sample
        # Calculate the sum of weights for each bin, i.e. the weighted spectrum
        output[i] = numba_weighted_histogram(data_sample, edges, data_eff)

    # Return the n generated Monte Carlo spectra
    return output


@njit(parallel=True)
def _mc_calibrated_spectrum_with_bkg(output, data, edges, bkg, calib, rng, n):
    # Determine number of measured data points
    data_counts = len(data[0])
    # Determine the number of background data points to draw
    bkg_sample_size = int(len(bkg[0]) / bkg[1] * data[1])
    # Define the edges for the background distribution
    xedges = np.linspace(0, 1, 513)
    # Calculate the background distribution
    bkg_pdf = numba_histogram(bkg[0], xedges)
    # Assign the background probabilities to the data points
    bkg_inds = np.digitize(data[0], xedges[1:-1])
    bkg_prob = bkg_pdf[bkg_inds]

    # Start the Monte Carlo simulation
    for i in prange(n):
        # Select data points based on Poisson sampling
        p = rng.poisson(lam=data_counts, size=1)[0]
        poisson_inds = rng.integers(0, data_counts, size=p)
        data_sample = data[0][poisson_inds]
        bkg_sample = bkg_prob[poisson_inds]
        # Remove data points based on the background distribution
        p = rng.poisson(lam=bkg_sample_size, size=1)[0]
        bkg_cdf = np.cumsum(bkg_sample)
        remove_inds = np.searchsorted(bkg_cdf, rng.random(p) * bkg_cdf[-1])
        data_sample = np.delete(data_sample, remove_inds)
        # Convert x values to wavelengths
        a0_sample = rng.normal(calib[0][0], calib[0][1], size=1)[0]
        a1_sample = rng.normal(calib[1][0], calib[1][1], size=1)[0]
        data_sample = a1_sample * data_sample + a0_sample
        # Calculate the sum of weights for each bin, i.e. the weighted spectrum
        output[i] = numba_histogram(data_sample, edges)

    # Return the n generated Monte Carlo spectra
    return output


@njit(parallel=True)
def _mc_calibrated_spectrum_with_bkg_qeff(output, data, edges, bkg, qeff, calib, rng, n):
    # Prepare the quantum efficiency correction
    n_eff = len(qeff[0])
    # Define the interpolation grid for the efficiencies
    xedges = np.linspace(0, 1, 513)
    xvalues = (xedges[1:] + xedges[:-1]) * 0.5
    # Assign the efficiencies to the data points
    eff_inds = np.digitize(data[0], xedges[1:-1])

    # Determine number of measured data points
    data_counts = len(data[0])
    # Determine the number of background data points to draw
    bkg_sample_size = int(len(bkg[0]) / bkg[1] * data[1])
    # Calculate the background distribution
    bkg_pdf = numba_histogram(bkg[0], xedges)
    # Assign the background probabilities to the data points
    bkg_prob = bkg_pdf[eff_inds]

    # Start the Monte Carlo simulation
    for i in prange(n):
        # Create a sample of the efficiencies
        eff_sample = np.ones(n_eff)
        for j in range(n_eff):
            eff_sample[j] = rng.normal(qeff[0][j], qeff[1][j], size=1)[0]
        # Interpolate the efficiencies to get a smoother spectrum
        eff_sample = np.interp(xvalues, qeff[2], eff_sample)
        # Get the efficiencies for each point
        data_eff = 1 / eff_sample[eff_inds]
        # Select data points based on Poisson sampling
        p = rng.poisson(lam=data_counts, size=1)[0]
        poisson_inds = rng.integers(0, data_counts, size=p)
        data_sample = data[0][poisson_inds]
        data_eff = data_eff[poisson_inds]
        bkg_sample = bkg_prob[poisson_inds]
        # Remove data points based on the background distribution
        p = rng.poisson(lam=bkg_sample_size, size=1)[0]
        bkg_cdf = np.cumsum(bkg_sample)
        remove_inds = np.searchsorted(bkg_cdf, rng.random(p) * bkg_cdf[-1])
        data_sample = np.delete(data_sample, remove_inds)
        data_eff = np.delete(data_eff, remove_inds)
        # Convert x values to wavelengths
        a0_sample = rng.normal(calib[0][0], calib[0][1], size=1)[0]
        a1_sample = rng.normal(calib[1][0], calib[1][1], size=1)[0]
        data_sample = a1_sample * data_sample + a0_sample
        # Calculate the sum of weights for each bin, i.e. the weighted spectrum
        output[i] = numba_weighted_histogram(data_sample, edges, data_eff)

    # Return the n generated Monte Carlo spectra
    return output


class Spectrum:
    """Fluorescence spectrum.

    """

    def __init__(self,
        xy: np.ndarray,
        time: int = None,
        **norm,
    ) -> None:
        """Initialize an instance of the Spectrum class from a 2D array
        containing x and y values of photon hits.

        In most cases it is recommended to use the `from_h5` class method
        to load the data from an h5 file generated by metro2hdf.

        Parameters
        ----------
        xy: np.ndarray
            2D array containing x and y values of photon hits.
        time: int, optional
            Measurement time in seconds for normalization. Default: None.
        **norm
            Additional normalization parameters as keyword arguments
            like the upstream intensity or target density. The values
            can be either floats or 1D arrays containing the value and
            its uncertainty.

        """
        # Store the passed data
        self._xy = xy
        self._t = time
        self._norm = list(norm.keys())
        for key, value in norm.items():
            setattr(self, key, value)

    @classmethod
    def from_h5(cls,
        file_path: str,
        anode: PositionAnode,
        raw: str = "dld_rd#raw",
        time: int = None,
        **norm,
    ) -> None:
        """Load a Spectrum from an h5 file generated by metro2hdf.

        Parameters
        ----------
        file_path: str
            Path to the h5 file.
        anode: PositionAnode
            Anode object to process the raw data.
        raw: str, optional
            Path to the raw data in the h5 file. Default:
            "dld_rd#raw".
        time: int, optional
            Measurement time in seconds for normalization. Default: None.
        **norm
            Additional normalization parameters as keyword arguments
            like `intensity_upstream` or `target_density`.

        Returns
        -------
        Spectrum
            The created Spectrum object.

        """
        with h5py.File(file_path, "r") as h5:
            # Load the raw data
            raw = np.asarray(h5[raw + "/0/0.0"])

            # Load normalization values
            for key, h5path in norm.items():
                if h5path not in h5:
                    raise ValueError("Normalization parameter not found.")

                if h5path.endswith("avg"):
                    norm[key] = h5[h5path + "/0"][0]

                else:
                    values = np.asarray(h5[h5path + "/0/0.0"])
                    norm[key] = np.array([np.mean(values), np.std(values)]).flatten()

        # Initialize the Spectrum
        return cls(anode.process(raw), time=time, **norm)

    def xy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the detector image of the spectrum.

        Returns
        -------
        np.ndarray
            The x and y values of the photon hits.

        """
        return self._xy[:,0], self._xy[:,1]

    def det_image(self,
        bins: Union[ArrayLike, Tuple[ArrayLike, ArrayLike]] = None,
        x_lim: Tuple[float, float] = (0, 1),
        y_lim: Tuple[float, float] = (0, 1),
        figsize: Tuple[float, float] = (6, 6),
        num: Union[str, int] = None,
        fig: Figure = None,
        ax: Tuple[Axes, Axes, Axes, Axes] = None,
    ) -> Tuple[Figure, Axes]:
        """Plot the detector image of the spectrum.

        Parameters
        ----------
        ax: Axes, optional
            Axes object to plot the detector image on. If not provided,
            a new figure is created.

        Returns
        -------
        Tuple[Figure, Axes]
            The figure and axes objects of the plot.

        """
        # Create the figure
        if fig is None or ax is None:
            fig = plt.figure(num=num, figsize=figsize, clear=True)

            # grid with columns=2, row=2
            gs = gridspec.GridSpec(
                2, 2, width_ratios=[3, 1], height_ratios=[1, 3],
                wspace=0.05, hspace=0.05
            )

            # 2d detector image is subplot 2: lower left
            ax_det = plt.subplot(gs[2])

            # x projection is subplot 0: upper left
            ax_x = plt.subplot(gs[0], sharex=ax_det)

            # y projection is subplot 3: lower right
            ax_y = plt.subplot(gs[3], sharey=ax_det)

            # colorbar is subplot 1: upper right
            ax_cb = plt.subplot(gs[1])
            ax_cb.axis("off")
            ax_cb_inset = ax_cb.inset_axes([0.0, 0.0, 0.25, 1.0])

            # Remove x and y tick labels
            ax_x.tick_params(axis='both', labelbottom=False)
            ax_y.tick_params(axis='both', labelleft=False)

            # Remove grid from the detector image and colorbar
            ax_det.grid(False)
            ax_cb_inset.grid(False)

        else:
            try:
                ax_det, ax_x, ax_y, ax_cb_inset = ax

                ax_det.clear()
                ax_x.clear()
                ax_y.clear()
                ax_cb_inset.clear()

            except:
                raise ValueError("Invalid axes sequence provided.")

        # Get the data
        x, y = self.xy()

        if bins is None:
            # Define x and y edges
            bins = np.histogram([], bins=512, range=(0, 1))[1]

        # Histogram the data
        hist_xy, x_edges, y_edges = np.histogram2d(x, y, bins=bins)

        # Define a meshgrid
        x_mesh, y_mesh = np.meshgrid(x_edges, y_edges)

        # Get a colormap from matplotlib
        cmap = plt.get_cmap("YlOrBr_r")

        # Get color for the projections
        color = cmap(0)

        # Set the lowest value to white
        colors = cmap(np.linspace(0, 1, cmap.N))
        colors[0] = (1, 1, 1, 1)

        # Create a colormap with white as the lowest value
        cmap = mcolors.ListedColormap(colors)

        # Plot the detector image
        pcm = ax_det.pcolormesh(
            x_mesh, y_mesh, hist_xy.T, cmap=cmap, rasterized=True
        )

        # Create a colorbar
        fig.colorbar(pcm, cax=ax_cb_inset)

        # Project the detector image onto the x and y axes
        hist_x = np.histogram(x, bins=x_edges)[0]
        hist_y = np.histogram(y, bins=y_edges)[0]

        # Plot the x and y projections
        ax_x.stairs(hist_x, x_edges, color=color)
        ax_y.stairs(hist_y, y_edges, color=color, orientation="horizontal")

        # Remove the first tick label of the x and y projection
        plt.setp(ax_x.get_yticklabels()[0], visible=False)
        plt.setp(ax_y.get_xticklabels()[0], visible=False)

        # Set the limits (this changes the positon of ax_det)
        ax_det.set_xlim(x_lim)
        ax_x.set_xlim(x_lim)
        ax_det.set_ylim(y_lim)
        ax_y.set_ylim(y_lim)

        # Set the labels
        ax_det.set_xlabel("x [arb. u.]")
        ax_det.set_ylabel("y [arb. u.]")

        return fig, (ax_det, ax_x, ax_y, ax_cb_inset)

    def counts(self,
        roi: Tuple[Tuple[float, float], Tuple[float, float]] = None,
        qeff: Tuple[np.ndarray, np.ndarray, np.ndarray] = None,
        bkg: Union[Spectrum, float] = None,
    ) -> Tuple[float, float]:
        """Get the number of counts in the spectrum and the estimated
        uncertainty.

        Parameters
        ----------
        roi: Tuple[Tuple[float, float], Tuple[float, float]], optional
            Region of interest for the detector in the form
            `((xmin, xmax), (ymin, ymax))`. If not provided, the
            full detector is used.
        qeff: Tuple[np.ndarray, np.ndarray, np.ndarray], optional
            Detector efficiencies in the form `(values, errors, x)`.
        bkg: Union[Spectrum, np.ndarray], optional
            Background spectrum (dark counts) to be subtracted that can
            be provided either as an instance of `Spectrum` or as an
            array of the length `len(edges) - 1`. For this to work
            properly, the background spectrum must be normalized to
            its measurement duration.

        Returns
        -------
        Tuple[float, float]
            The number of counts (normalized) and the respective
            uncertainty.

        """
        data = np.copy(self._xy)

        # Use the full detector if roi not provided
        if roi is not None:
            # Apply y roi filter
            data = data[data[:,1] > roi[1][0]]
            data = data[data[:,1] < roi[1][1]]
            # Discard y values
            data = data[:,0].flatten()
            # Apply x roi filter
            data = data[data > roi[0][0]]
            data = data[data < roi[0][1]]

        # Apply spatial detector efficiency correction
        if qeff is not None:
            eff_val, eff_err, eff_x = qeff
            xedges = (eff_x[:-1] + eff_x[1:]) * 0.5
            eff_inds = np.digitize(data, xedges)

            def sum_weights(eff):
                return np.sum(1 / eff[eff_inds])

            # Sum the efficiencies and propagate the uncertainties
            counts, error = propagate(sum_weights, eff_val, eff_err**2)
            error = np.sqrt(np.diag(error))[0]

            # Include the Poisson uncertainties
            error = np.sqrt(len(data) + error**2)

        else:
            # Calculate the number of counts and the Poisson uncertainty
            counts = len(data)
            error = np.sqrt(counts)

        # Normalize data to measurement duration
        if self._t is not None:
            counts /= self._t
            error /= self._t

        # Subtract background before further normalization
        if isinstance(bkg, Spectrum):
            bkg_counts, bkg_err = bkg.counts(roi=roi, qeff=qeff, bkg=None)
            # Using just the statistical uncertainty of the background
            # counts would underestimate the uncertainty of the subtraction
            bkg_err = np.sqrt(bkg_counts * self._t) / self._t
            counts = max(counts - bkg_counts, 0)
            error = np.sqrt(error**2 + bkg_err**2)

        # Normalize data to account for beam intensity, gas
        # pressure, etc.
        for norm in self._norm:
            if isinstance(getattr(self, norm), np.ndarray):
                val, err = getattr(self, norm)
                error = np.sqrt(error**2 / val**2 + err**2 * counts**2 / val**4)
                counts /= val
            else:
                counts /= getattr(self, norm)
                error /= getattr(self, norm)

        # Return the counts and the uncertainty
        return counts, error

    def spectrum(self,
        edges: np.ndarray,
        roi: Tuple[Tuple[float, float], Tuple[float, float]] = None,
        qeff: Tuple[np.ndarray, np.ndarray, np.ndarray] = None,
        bkg: Union[Spectrum, np.ndarray] = None,
        calib: Tuple[Tuple[float, float], Tuple[float, float]] = None,
        err_prop: Literal["montecarlo", "none"] = "montecarlo",
        mc_samples: int = 10000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the spectum and its uncertainties for a given
        set of bin edges.

        Parameters
        ----------
        edges: np.ndarray
            Bin edges for the histogram. For a calibrated spectrum,
            these should be in wavelength units. For an uncalibrated
            spectrum, these should be between 0 and 1.
        roi: Tuple[Tuple[float, float], Tuple[float, float]], optional
            Region of interest for the detector in the form
            `((xmin, xmax), (ymin, ymax))`. If not provided, the
            full detector is used.
        qeff: Tuple[np.ndarray, np.ndarray, np.ndarray], optional
            Detector efficiencies in the form `(values, errors, x)`.
            The efficiencies are interpolated to 512 points between
            0 and 1.
        bkg: Union[Spectrum, np.ndarray], optional
            Background spectrum (dark counts) to be subtracted that can
            be provided either as an instance of `Spectrum` or as an
            array of the length `len(edges) - 1`. For this to work
            properly, the background spectrum must be normalized to
            its measurement duration.
        calib: Tuple[Tuple[float, float], Tuple[float, float]], optional
            Wavelength calibration parameters in the form
            `((a0, err), (a1, err))`, where `a0` and `a1`
            correspond to wavelength = a1 * x + a0 and `err` to the
            respective uncertainties.
        err_prop: Literal["jacobi", "montecarlo", "none"], optional
            Error propagation method for handling the uncertainties of
            the efficiencies and the wavelength calibration. If
            `qeff = None` and `calib = None`, this setting has no
            effect. Can be 'montecarlo', or 'none'.
        mc_samples: int, optional
            Number of Monte Carlo samples to use for error propagation.
            Has no effect if `err_prop = 'none'`.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The spectrum and its uncertainties.
        
        """
        # Get x and y values of the photon hits
        data = np.copy(self._xy)

        # Define the roi as the full detector if not provided
        if roi is None:
            roi = ((0, 1), (0, 1))
        # Apply y roi filter
        data = data[data[:,1] > roi[1][0]]
        data = data[data[:,1] < roi[1][1]]
        # Don't need y values anymore
        data = data[:,0].flatten()

        # Prepare the background subtraction
        if isinstance(bkg, Spectrum):
            # Get the measurement duration of the background
            bkg_t = bkg._t
            # Test if background and data both have a time value
            if bkg_t is None or self._t is None:
                raise ValueError("Both background and data must have a time value.")

            # Get the x and y values of the background spectrum
            bkg_data = bkg._xy

            # Apply y roi filter
            bkg_data = bkg_data[bkg_data[:,1] > roi[1][0]]
            bkg_data = bkg_data[bkg_data[:,1] < roi[1][1]]
            # Don't need y values anymore
            bkg_data = bkg_data[:,0].flatten()

            # Prepare the background to be passed to the spectrum calculation
            bkg = (bkg_data, bkg_t)

        else:
            # Continue without background subtraction
            bkg = None

        # Prepare the wavelength calibration parameters
        if calib is not None:
            try:
                calib_params = np.array(calib, dtype=np.float64)
                assert calib_params.shape == (2, 2)
            
            except:
                raise ValueError("Calibration parameters must be provided as ((a0, err), (a1, err)).")

            # Adjust x roi filter to wavelength binning
            wl_min = calib_params[1][0] * roi[0][0] + calib_params[0][0]
            wl_max = calib_params[1][0] * roi[0][1] + calib_params[0][0]
            roi = np.argwhere((edges < wl_min) | (edges > wl_max)).flatten()

        else:
            # Define dummy calibration parameters
            calib_params = np.array([[0, 0], [1, 0]])

            # Adjust x roi filter to binning
            roi = np.argwhere((edges < roi[0][0]) | (edges > roi[0][1])).flatten()

        # Prepare spatial detector efficiency correction
        if qeff is not None:
            try:
                # Test if the efficiencies are provided correctly
                qeff_val = np.array(qeff[0])
                qeff_err = np.array(qeff[1])
                qeff_x = np.array(qeff[2])
                assert len(qeff_val) == len(qeff_err)
                assert len(qeff_val) == len(qeff_x)

            except:
                raise ValueError("Detector efficiencies must be provided as (values, errors, x).")

            qeff = (qeff_val, qeff_err, qeff_x)

        if calib is None and qeff is None and bkg is None:
            spectrum = np.histogram(data, bins=edges)[0]
            spectrum = np.array(spectrum, dtype=np.float64)
            errors = np.sqrt(spectrum)

        elif err_prop == "montecarlo":
            # Initialize the random number generator
            rng = np.random.default_rng()

            # Initialize array for storing the sample results
            spectrum = np.zeros((mc_samples, len(edges) - 1), dtype=np.float64)

            # Perform the Monte Carlo simulation
            if qeff is None and bkg is None:
                spectrum = _mc_calibrated_spectrum(spectrum, data, edges, calib_params, rng, mc_samples)
            elif qeff is None:
                spectrum = _mc_calibrated_spectrum_with_bkg(
                    spectrum, (data, self._t), edges, bkg, calib_params, rng, mc_samples)
            elif bkg is None:
                spectrum = _mc_calibrated_spectrum_with_qeff(
                    spectrum, data, edges, qeff, calib_params, rng, mc_samples)
            else:
                spectrum = _mc_calibrated_spectrum_with_bkg_qeff(
                    spectrum, (data, self._t), edges, bkg, qeff, calib_params, rng, mc_samples)

            # Calculate mean and standard deviation of the sampled spectra
            errors = np.std(spectrum, ddof=1, axis=0)
            spectrum = np.mean(spectrum, axis=0)

        elif err_prop == "none":
            # Calibrate the data
            if calib is not None:
                data = calib_params[1][0] * data + calib_params[0][0]

            # Histogram the data without the efficiencies
            spectrum = np.histogram(data, bins=edges)[0]
            errors = np.sqrt(spectrum)

            # Interpolate and assign the efficiencies
            if qeff is not None:
                xedges = np.linspace(0, 1, 513)
                xvalues = (xedges[1:] + xedges[:-1]) * 0.5
                qeff = np.interp(xvalues, qeff[2], qeff[0])
                eff_inds = np.digitize(data, xedges[1:-1])
                qeff = 1 / qeff[eff_inds]

                # Histogram the data with the efficiencies
                weights = np.histogram(data, bins=edges, weights=qeff)[0]

                # Calculate the uncertainties
                nonzero = spectrum > 0
                errors[nonzero] = weights[nonzero] * np.sqrt(2 / spectrum[nonzero])
                spectrum = weights

            if bkg is not None:
                # Calibrate the background data
                if calib is not None:
                    bkg_data = calib_params[1][0] * bkg_data + calib_params[0][0]

                # Histogram the background data
                if qeff is not None:
                    # Interpolate and assign the efficiencies
                    bkg_inds = np.digitize(bkg_data, xedges[1:-1])
                    bkg_qeff = 1 / qeff[bkg_inds]

                    # Histogram the background data with the efficiencies
                    bkg = np.histogram(bkg_data, bins=edges, weights=bkg_qeff)[0]

                else:
                    # Histogram the background data without the efficiencies
                    bkg = np.histogram(bkg_data, bins=edges)[0]

                # Convert to floats
                bkg = np.array(bkg, dtype=np.float64)

                # Normalize the background to the measurement duration
                bkg = bkg / bkg_t * self._t

                # Subtract the background
                spectrum -= bkg
                spectrum[spectrum < 0] = 0

        else:
            raise ValueError("Error propagation method must be 'montecarlo' or 'none'.")

        # Normalize data to measurement duration per step
        if self._t is not None:
            spectrum /= self._t
            errors /= self._t

        # Normalize data to account for beam intensity, gas 
        # pressure, etc.
        for norm in self._norm:
            if isinstance(getattr(self, norm), np.ndarray):
                val, err = getattr(self, norm)
                errors = np.sqrt(errors**2 / val**2 + err**2 * spectrum**2 / val**4)
                spectrum /= val
            else:
                spectrum /= getattr(self, norm)
                errors /= getattr(self, norm)

        # Apply x roi filter
        spectrum[roi[:-1]] = 0
        errors[roi[:-1]] = 0

        # Return the spectrum and uncertainties
        return spectrum, errors

    def transform_norm(self, norm: str, func: callable):
        """Transform the specified normalization values using a given
        function.

        Parameters
        ----------
        norm: str
            Name of the normalization parameter to transform.
        func: callable
            Function to apply to the normalization values. The function
            should take a single argument of type float and return a
            float.

        """
        val = getattr(self, norm)

        if isinstance(val, np.ndarray):
            val, err = propagate(func, val[0], val[1]**2)
            setattr(self, norm, np.array([val, np.sqrt(err)]))

        else:
            setattr(self, norm, func(val))

    def convert_unit(self, norm: str, fro: str, to: str):
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

        except ImportError:
            raise ImportError("pint is required to convert units.")

        ureg = UnitRegistry()
        # Convert the normalization values
        self.transform_norm(norm, lambda x: ureg.Quantity(x, fro).m_as(to))


class BaseScan:

    def __init__(self,
        data_files: Sequence[str],
        anode: PositionAnode,
        scan_var: str = None,
        raw: str = "dld_rd#raw",
        time_per_step: Union[int, Sequence[int]] = None,
        roi: Tuple[Tuple[float, float], Tuple[float, float]] = None,
        **norm,
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
            spec, steps = self._load_spectra(f, scan_var, raw, anode, t, **norm)
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
                spectra.append(
                    Spectrum(anode.process(data), time=time_per_step, **step_norm)
                )

        # Return the spectra and energies
        return spectra, step_val

    @property
    def spectra(self) -> np.ndarray:
        return self._spectra

    @spectra.setter
    def spectra(self, spectra: np.ndarray) -> None:
        raise AttributeError("Attribute 'spectra' is read-only.")

    @property
    def steps(self) -> np.ndarray:
        return self._steps

    @steps.setter
    def steps(self, steps: np.ndarray) -> None:
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
    def id(self) -> np.ndarray:
        return self._id

    @id.setter
    def id(self, id: np.ndarray) -> None:
        raise AttributeError("Attribute 'id' is read-only.")

    @property
    def roi(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return self._roi

    @roi.setter
    def roi(self, roi: Tuple[Tuple[float, float], Tuple[float, float]]) -> None:
        self._roi = self.prepare_roi(roi)

    def prepare_roi(self,
        roi: Tuple[Tuple[float, float], Tuple[float, float]],
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
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

            except:
                raise ValueError(
                    "Region of interest must be provided as "
                    "((xmin, xmax), (ymin, ymax))."
                )

            return roi

    def select_step_range(self,
        step_range: Tuple[float, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if step_range is not None:
            try:
                mask = (self.steps >= step_range[0]) & (self.steps <= step_range[1])
                steps = self.steps[mask]
                spectra = self.spectra[mask]

            except:
                raise ValueError("step_range must be a tuple of two floats.")

            return spectra, steps

        else:
            return self.spectra, self.steps

    def norm(self,
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

        except ImportError:
            raise ImportError("pint is required to convert units.")

        ureg = UnitRegistry()
        trafo = lambda x: ureg.Quantity(x, fro).m_as(to)
        self.transform_norm(norm, trafo)

    def remove_steps(self,
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

    def __init__(self,
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
    def calib(self, calib: Tuple[Tuple[float, float], Tuple[float, float]]) -> None:
        self._calib = self.prepare_calib(calib)

    def prepare_qeff(self,
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

        except:
            raise ValueError("qeff could not be parsed.")

        return values, errors, x

    def prepare_bkg(self,
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

    def prepare_calib(self,
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

        except:
            raise ValueError("calib could not be parsed.")

        return calib_params

    def counts(self,
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

    def spectrum_at(self,
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
            edges, roi=roi, qeff=qeff, bkg=bkg, calib=calib, err_prop=err_prop,
            mc_samples=mc_samples
        )
    
    def spectrum_at_step(self,
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
            idx, edges, roi=roi, qeff=qeff, bkg=bkg, calib=calib,
            err_prop=err_prop, mc_samples=mc_samples
        )

    def show_spectra(self):
        """Plot the spectra in an interactive window.

        """
        from agepy.interactive import run
        from agepy.spec.interactive.photons_scan import SpectrumViewer

        # Intialize the viewer
        mw = SpectrumViewer(self)

        # Run the application
        run(mw)


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
    roi: Tuple[Tuple[float, float], Tuple[float, float]], optional
        Region of interest for the detector in the form
        `((xmin, xmax), (ymin, ymax))`. Default: None.
    **norm
        Normalization parameters as keyword arguments with the name as
        key and the h5 path as the value. For example,
        `intensity_upstream = mirror#value`.

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
        energy_uncertainty: Union[np.ndarray, float],
        energies: str = None,
        raw: str = "dld_rd#raw",
        time_per_step: Union[int, Sequence[int]] = None,
        roi: Tuple[Tuple[float, float], Tuple[float, float]] = None,
        **norm,
    ) -> None:
        super().__init__(
            data_files, anode, energies, raw, time_per_step, roi, **norm
        )

        # Set attributes
        self.energy_uncertainty = energy_uncertainty
        self._phex_assignments = None
        self._phex_label = None
        self._phem_assignments = None
        self._phem_label = None

    @property
    def energies(self) -> np.ndarray:
        return self._steps

    @energies.setter
    def energies(self, value: np.ndarray) -> None:
        self.steps = value

    @property
    def energy_uncertainty(self) -> np.ndarray:
        return self._energy_uncertainty

    @energy_uncertainty.setter
    def energy_uncertainty(self, value: Union[np.ndarray, float]) -> None:
        if value is None:
            self._energy_uncertainty = None
            return

        if isinstance(value, np.ndarray):
            if value.shape == (len(self.spectra),):
                self._energy_uncertainty = value

            else:
                raise ValueError(
                    "Uncertainties must be of the same length as spectra."
                )

        elif isinstance(value, (int, float)):
            self._energy_uncertainty = np.full(
                len(self._steps), value, dtype=np.float64
            )

        else:
            raise ValueError("Uncertainties must be a numpy array or a float.")

    def remove_steps(self,
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
        # Call the base class method
        mask = super().remove_steps(measurement_number, steps)

        # Remove the uncertainties
        self._energy_uncertainty = np.delete(self._energy_uncertainty, mask)

        return mask

    def select_by_phex(self,
        phex: Dict[str, Union[str, int]],
        n_std: int = 1,
        ignore_overlap: bool = False,
    ) -> Tuple[int, NDArray]:
        # Find the phex assignment
        df = self._phex_assignments.copy()
        for key, value in phex.items():
            df.query(f"{key} == @value", inplace=True)

            # Check if there are matching assignments
            if df.empty:
                raise ValueError("Phex assignment not found.")

        # At this point there should be only one assignment
        assert len(df) == 1, "Multiple or no assignments found."

        # Get the fit results
        fit_val = df["val"].iloc[0]
        fit_err = df["err"].iloc[0]

        # Select energy steps within n_std standard deviations of the mean
        step_idx = np.argwhere(
            np.abs(self.steps - fit_val[1]) < fit_val[2] * n_std
        ).flatten()

        # Check if steps were found
        if len(step_idx) == 0 or ignore_overlap:
            return df.index[0], step_idx

        # Define energy range
        e_range = (
            fit_val[1] - fit_val[2] * n_std, fit_val[1] + fit_val[2] * n_std
        )

        # Check if multiple phex assignments overlap
        overlap = self._phex_assignments.query(
            "E > @e_range[0] and E < @e_range[1]"
        )

        for i, row in overlap.iterrows():
            if i == df.index[0]:
                continue

            overlap_val = row["val"]
            overlap_idx = np.argwhere(
                np.abs(self.steps - overlap_val[1]) < overlap_val[2]
            ).flatten()

            # Remove the overlapping steps
            overlap_idx = np.setdiff1d(step_idx, overlap_idx)

            # Check if steps remain
            if len(overlap_idx) == 0:
                warnings.warn("No steps found without overlap.")

            else:
                step_idx = overlap_idx

        return df.index[0], step_idx

    def assign_phex(self,
        reference: pd.DataFrame,
        label: Dict[str, Union[Sequence[str], int]],
        energy_range: float,
    ) -> int:
        from agepy.interactive import get_qapp
        from agepy.spec.interactive.photons_phex import AssignPhex

        # Get the Qt application
        app = get_qapp()

        # Intialize the viewer
        mw = AssignPhex(self, reference, label, energy_range)
        mw.show()

        # Run the application
        app.exec()

    def save_phex(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self._phex_assignments, f)

    def load_phex(self, path: str) -> None:
        with open(path, "rb") as f:
            self._phex_assignments = pickle.load(f)

    def eval_phex(self,
        reference: pd.DataFrame,
        plot: bool = True,
        plot_pulls: bool = True,
    ) -> Tuple[plt.Figure, plt.Axes]:
        # Create a list of fit values and reference values
        fit_energies = []
        ref_energies = []
        labels = []

        # Find matching transitions
        for i, row in self._phex_assignments.iterrows():
            ref = reference.copy()

            # Look for the excitation in the reference data
            label = ""
            for key, value in row.items():
                # Skip the fit values
                if key in ["E", "val", "err"]:
                    continue

                # Check if matching entries remain
                if ref.empty:
                    continue

                ref.query(f"{key} == @value", inplace=True)

                label += f"{key} = {value}, "

            # Check if matching entries remain
            if ref.empty:
                continue

            # Remove the last comma and space
            label = label[:-2]

            # Append the found match
            fit_energies.append([row["val"][1], row["err"][1]])
            ref_energies.append(ref["E"].iloc[0])
            labels.append(label)

        # Convert to numpy arrays
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

        # Try to import iminuit for fitting
        try:
            from iminuit import Minuit
            from iminuit.cost import LeastSquares

        except ImportError:
            raise ImportError("iminuit is required for fitting.")

        # Define the cost function
        c = LeastSquares(ref_energies, fit_energies[:,0], fit_energies[:,1], model)

        # Initialize the minimizer with starting values
        m = Minuit(c, b1=1, b0=0)

        # Set limits
        m.limits["b1"] = (0.9, 1.1)
        m.limits["b0"] = (-0.1, 0.1)

        # Perform the fit
        m.migrad()

        # Check for bad assignments
        pulls = np.abs(c.pulls(m.values))
        inds = np.argwhere(pulls > 5).flatten()
        for idx in inds:
            print(
                "Bad assignment of:", labels[idx], "with diff:",
                c.prediction(m.values)[idx] - fit_energies[idx,0]
            )

        # Plot the results
        if plot:
            # Create the figure
            fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={"hspace": 0.3})

            # Plot the assignments
            ax[0].errorbar(
                ref_energies, fit_energies[:,0], yerr=fit_energies[:,1],
                fmt="s", markersize=1.5, label="Assign. Phex"
            )

            # Plot the fit results
            ax[0].plot(ref_energies, c.prediction(m.values), label="Lin. Regr.")

            # Plot the legend with the chi2/ndof
            chi2ndof = m.fmin.reduced_chi2
            ax[0].legend(
                title=r"$\chi^2\;/\;$ndof = " + f"{chi2ndof:.2f}", loc="best"
            )

            # Set title and labels
            ax[0].set_title(r"Assigned Photon-Excitation Energies $E_\text{data}$")
            ax[0].set_ylabel(r"$E_\text{data}$ [eV]")
            ax[1].set_xlabel(r"$E_\text{literature}$ [eV]")

            # Plot the residuals
            ax[1].axhline(0, color="black", linestyle="--", alpha=0.9)
            if plot_pulls:
                # Plot the pulls
                ax[1].step(ref_energies, c.pulls(m.values), where="mid")

                # Set title and label
                ax[1].set_title("Studentized Residuals of the Linear Regression")
                ax[1].set_ylabel("Pulls")

            else:
                # Plot the differences of the fit to the data
                ax[1].step(ref_energies, c.prediction(m.values) - fit_energies[:,0], where="mid")

                # Set title and label
                ax[1].set_title("Difference to the Linear Regression")
                ax[1].set_ylabel(r"$(\text{Lin. Regr.} - E_\text{data})$ [eV]")

            return fig, ax

        return None, None

    def assign_phem(self,
        reference: pd.DataFrame,
        label: Dict[str, Union[Sequence[str], int]],
        calib_guess: Tuple[float, float],
        bins: int = 512,
    ) -> None:
        from agepy.interactive import get_qapp
        from agepy.spec.interactive.photons_phem import AssignPhem

        # Create the bin edges
        edges = np.histogram([], bins=bins, range=(0, 1))[1]

        # Get the Qt application
        app = get_qapp()

        # Intialize the viewer
        mw = AssignPhem(self, edges, reference, label, calib_guess)
        mw.show()

        # Run the application
        app.exec()

    def save_phem(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self._phem_assignments, f)

    def load_phem(self, path: str) -> None:
        with open(path, "rb") as f:
            self._phem_assignments = pickle.load(f)

    def eval_phem(self,
        reference: pd.DataFrame,
        calib_guess: Tuple[float, float],
        plot: bool = True,
        plot_pulls: bool = True,
    ) -> Tuple[plt.Figure, plt.Axes]:
        # Create a list of fit values and reference values
        fit_wl = []
        ref_wl = []
        labels = []

        # Find matching transitions
        for i, row in self._phem_assignments.iterrows():
            ref = reference.copy()

            # Look for the transition in the reference data
            label = ""
            for key, value in row.items():
                # Skip not existing keys
                if key not in ref.columns:
                    continue

                ref.query(f"{key} == @value", inplace=True)

                # Check if matching entries remain
                if ref.empty:
                    continue

                label += f"{key} = {value}, "

            # Remove the last comma and space
            label = label[:-2]

            # Get the fit result
            idx = row["fit"].par.index("loc")
            val = row["fit"].val[idx]
            err = row["fit"].err[idx]

            # Append the found match
            fit_wl.append([val, err])
            ref_wl.append(ref["E"].iloc[0])
            labels.append(label)

        # Convert to numpy arrays
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

        # Try to import iminuit for fitting
        try:
            from iminuit import Minuit
            from iminuit.cost import LeastSquares

        except ImportError:
            raise ImportError("iminuit is required for fitting.")

        # Define the cost function
        c = LeastSquares(ref_wl, fit_wl[:,0], fit_wl[:,1], model)

        # Parse start values
        b1start = 1 / calib_guess[1]
        b0start = -calib_guess[0] / calib_guess[1]

        # Initialize the minimizer with starting values
        m = Minuit(c, b1=b1start, b0=b0start)

        # Perform the fit
        m.migrad()

        # Get the calibration results
        b0, b1 = m.values["b0"], m.values["b1"]
        b0err, b1err = m.errors["b0"], m.errors["b1"]

        # Convert to calibration parameters
        a1 = 1 / b1
        a1err = b1err / b1**2
        a0 = -b0 / b1
        a0err = np.sqrt(b0err**2 / b1**2 + b0**2 * b1err**2 / b1**4)

        # Set the calibration
        self.calib = ((a0, a0err), (a1, a1err))

        # Check for bad assignments
        pulls = np.abs(c.pulls(m.values))
        inds = np.argwhere(pulls > 5).flatten()
        for idx in inds:
            print(
                "Bad assignment of:", labels[idx], "with diff:",
                c.prediction(m.values)[idx] - fit_wl[idx,0]
            )

        # Plot the results
        if plot:
            # Create the figure
            fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={"hspace": 0.3})

            # Plot the assignments
            ax[0].errorbar(
                ref_wl, fit_wl[:,0], yerr=fit_wl[:,1],
                fmt="s", markersize=1.5, label="Assign. Phem"
            )

            # Plot the fit results
            ax[0].plot(ref_wl, c.prediction(m.values), label="Lin. Regr.")

            # Plot the legend with the chi2/ndof
            chi2ndof = m.fmin.reduced_chi2
            ax[0].legend(
                title=r"$\chi^2\;/\;$ndof = " + f"{chi2ndof:.2f}", loc="best"
            )

            # Set title and labels
            ax[0].set_title(r"Assigned Detector Positions $x_\text{data}$")
            ax[0].set_ylabel(r"$x_\text{data}$ [arb. u.]")
            ax[1].set_xlabel(r"$\lambda_\text{literature}$ [nm]")

            # Plot the residuals
            ax[1].axhline(0, color="black", linestyle="--", alpha=0.9)
            if plot_pulls:
                # Plot the pulls
                ax[1].step(ref_wl, c.pulls(m.values), where="mid")

                # Set title and label
                ax[1].set_title("Studentized Residuals of the Linear Regression")
                ax[1].set_ylabel("Pulls")
            else:
                # Plot the differences of the fit to the data
                ax[1].step(ref_wl, c.prediction(m.values) - fit_wl[:,0], where="mid")

                # Set title and label
                ax[1].set_title("Difference to the Linear Regression")
                ax[1].set_ylabel(r"$(\text{Lin. Regr.} - x_\text{data})$ [arb. u.]")

            return fig, ax

        return None, None

    def save_calib(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.calib, f)

    def load_calib(self, path: str) -> None:
        with open(path, "rb") as f:
            self.calib = pickle.load(f)

    def assigned_spectrum(self,
        phex: Dict[str, Union[str, int]],
        edges: np.ndarray,
        n_std: float = 1.0,
        normalize: bool = False,
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
        phex: Dict[str, Union[str, int]]
            Dictionary specifying a photon excitation. The photn
            excitation needs to be in the assignments made with
            `EnergyScan.assign_phex`.
        edges: np.ndarray
            Bin edges for the histogram. For a calibrated spectrum,
            these should be in wavelength units. For an uncalibrated
            spectrum, these should be between 0 and 1.
        n_std: float, optional
            Number of standard deviations of the assignment fit to use
            for selecting spectra.
        normalize: bool, optional
            Whether to normalize the spectra to their excitation.
            This will increase the uncertainty of the spectrum, because
            of the excitation fit uncertainty.
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
        # Find the phex assignment
        phex_idx, step_idx = self.select_by_phex(phex, n_std)

        if len(step_idx) == 0:
            warnings.warn("Trying with 2 * n_std.")
            phex_idx, step_idx = self.select_by_phex(phex, 2 * n_std)

        if len(step_idx) == 0:
            warnings.warn("Ignore overlapping assignments.")
            phex_idx, step_idx = self.select_by_phex(
                phex, n_std, ignore_overlap=True
            )

        if len(step_idx) == 0:
            warnings.warn("No steps found for the given photon excitation.")
            return np.zeros(len(edges) - 1), np.zeros(len(edges) - 1)

        # Get the fit results
        fit_val = self._phex_assignments.loc[phex_idx, "val"]
        fit_err = self._phex_assignments.loc[phex_idx, "err"]

        # Initialize arrays for the summed spectrum
        spectrum = np.zeros(len(edges) - 1)
        errors = np.zeros(len(edges) - 1)

        # Combine the spectra
        for idx in step_idx:
            # Calculate the spectrum
            spec, err = self.spectrum_at(
                idx, edges, roi=roi, qeff=qeff, bkg=bkg, calib=calib,
                err_prop=err_prop, mc_samples=mc_samples
            )

            # Normalize with the excitation fit results
            if normalize:
                # Define function for evaluating the excitation
                def calc_exc(par):
                    y = norm.pdf(self.steps[idx], par[0], par[1])
                    y_max = norm.pdf(par[0], par[0], par[1])
                    return y / y_max

                # Evaluate the excitation at the step value
                exc, exc_err = propagate(calc_exc, fit_val[1:], fit_err[1:]**2)
                exc_err = np.sqrt(exc_err)

                # Normalize the spectrum
                spec /= exc
                err = np.sqrt(
                    err**2 / exc**2 + spec**2 * exc_err**2 / exc**4
                )

            # Add the spectrum to the sum
            spectrum += spec
            errors += err**2

        # Get Gaussian error (currently correlations are not considered)
        errors = np.sqrt(errors)

        # Normalize to the number of summed spectra
        if normalize:
            spec /= len(step_idx)
            err /= len(step_idx)

        # Return the summed spectrum
        return spectrum, errors

    def phexphem(self,
        xedges: np.ndarray = None,
        yedges: np.ndarray = None,
        roi: Tuple[Tuple[float, float], Tuple[float, float]] = None,
        qeff: bool = True,
        bkg: bool = True,
        calib: bool = True,
    ) -> np.ndarray:
        """
        
        """
        # Parse the given calculation options
        roi = self.prepare_roi(roi)
        qeff = self.prepare_qeff(qeff)
        bkg = self.prepare_bkg(bkg)
        calib = self.prepare_calib(calib)

        # Prepare the x edges
        if xedges is None:
            dx = np.mean(np.diff(self.steps) * 0.5)
            dec = -int(np.floor(np.log10(dx)))
            dx = round(dx, dec)
            xmin = round(np.min(self.steps), dec) - dx
            xedges = np.arange(xmin, np.max(self.steps) + dx * 2, dx * 2)

        # Prepare the y edges
        if yedges is None:
            yedges = np.histogram([], bins=512, range=(0, 1))[1]

        if calib is not None:
            yedges = calib[1][0] * yedges + calib[0][0]

        # Create an empty map
        hist = np.zeros((len(yedges) - 1, len(xedges) - 1))

        # Fill the map
        weights, _ = np.histogram(self.steps, bins=xedges)
        inds = np.digitize(self.steps, xedges[1:-1])

        for spec, idx in zip(self.spectra, inds):
            hist[:, idx] += spec.spectrum(
                yedges,
                roi=self.roi,
                qeff=qeff,
                bkg=bkg,
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
        **norm,
    ) -> None:
        # Force the x roi to cover the full detector
        if roi is not None:
            roi[0][0] = 0
            roi[0][1] = 1

        # Load and process data
        super().__init__(data_files, anode, None, raw, time_per_step, roi, **norm)

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

    def interpolate(self,
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

    def interactive(self, bins: int = 512, mc_samples=10000) -> int:
        """Plot the spectra in an interactive window.

        """
        from agepy.interactive import get_qapp
        from agepy.spec.interactive.photons_qeff import EvalQEff

        # Create the edges
        edges = np.histogram([], bins=bins, range=(0, 1))[1]

        # Get the Qt application
        app = get_qapp()

        # Intialize the viewer
        mw = EvalQEff(self, edges, mc_samples)
        mw.show()

        # Run the application
        return app.exec()

    def plot(self,
        ax: Axes = None,
        color: str = "k",
        label: str = None
    ) -> Tuple[Figure, Axes]:
        """Plot the calculated detector efficiencies.

        """
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

    def __init__(self,
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
        """Plot the spectra in an interactive window.

        """
        from agepy.interactive import get_qapp
        from agepy.spec.interactive.photons_rot import EvalRot

        # Get the Qt application
        app = get_qapp()

        # Intialize the viewer
        mw = EvalRot(self, angle, offset)
        mw.show()

        # Run the application
        return app.exec()
