"""Processing and analysis of fluorescence spectra."""

from __future__ import annotations

from dataclasses import dataclass
import os
import numpy as np
from numba import njit, prange
from jacobi import propagate
from matplotlib import gridspec
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import h5py

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike, NDArray
    from agepy.spec.anodes import PositionAnode


@dataclass(frozen=True)
class Spectrum:
    """Fluorescence spectrum recorded with a position-sensitive
    detector stored as event (photon hit) delay-line timings.

    Parameters
    ----------
    raw: np.ndarray, shape (N,M)
        Array containing raw delay-line values of photon hits.
    time: int, optional
        Measurement time in seconds for normalization.
    norm: dict
        Additional normalization parameters as combinations of
        names and either an array of measured values or their
        average.

    """

    raw: NDArray
    time: int = 1
    norm: dict[str, NDArray | float] = {}
    step: str = "0.0"
    scan: str = "0"

    def xy(self, anode: PositionAnode) -> NDArray:
        """Get the event coordinates (x, y).

        Parameters
        ----------
        anode: PositionAnode
            Anode instance with a `process` method that converts raw values
            timing values from the used anode to x, y coordinates.

        Returns
        -------
        xy: np.ndarray, shape (N,2)
            The x and y values of the photon hits.

        """
        return anode.process(self.raw)

    def det_image(
        self,
        anode: PositionAnode,
        bins: int | ArrayLike = 512,
        x_lim: tuple[float, float] = (0, 1),
        y_lim: tuple[float, float] = (0, 1),
        figsize: tuple[float, float] = (6.4, 6.4),
        num: str | int | None = None,
        fig: Figure | None = None,
        ax: tuple[Axes, Axes, Axes, Axes] | None = None,
    ) -> tuple[Figure, tuple[Axes, Axes, Axes, Axes]]:
        """Plot the detector image of the spectrum.

        Parameters
        ----------
        anode: PositionAnode
            Anode instance with a `process` method that converts raw values
            timing values from the used anode to x, y coordinates.
        bins: int or array_like or [int, int] or [array, array], optional
            See `numpy.histogram2d`.
        x_lim: [float, float], optional
            Set plot limits on the image x axis and x projection.
        y_lim: [float, float], optional
            Set plot limits on the image y axis and y projection.
        figsize: [float, float], optional
            Figure size in inches.
        num: str or int, optional
            See `matplotlib.pyplot.figure`.
        figure: Figure, optional
            Pass a figure and axes to plot on. If `None` a new
            figure and axes are created.
        ax: [Axes, Axes, Axes, Axes], optional
            Pass a figure and axes to plot on. If `None` a new
            figure and axes are created.

        Returns
        -------
        fig: Figure
            See `matplotlib.figure.Figure`.
        ax: [Axes, Axes, Axes, Axes]
            See `matplotlib.axes.Axes`.

        """
        # Create the figure
        if fig is None or ax is None:
            fig = plt.figure(num=num, figsize=figsize, clear=True)

            # grid with columns=2, row=2
            gs = gridspec.GridSpec(
                2,
                2,
                width_ratios=[3, 1],
                height_ratios=[1, 3],
                wspace=0.05,
                hspace=0.05,
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
            ax_x.tick_params(axis="both", labelbottom=False)
            ax_y.tick_params(axis="both", labelleft=False)

            # Remove grid from the detector image and colorbar
            ax_det.grid(False)
            ax_cb_inset.grid(False)

        else:
            # Use given axes
            ax_det, ax_x, ax_y, ax_cb_inset = ax

            # Clear the given axes before plotting
            ax_det.clear()
            ax_x.clear()
            ax_y.clear()
            ax_cb_inset.clear()

        # Get the data
        xy = self.xy(anode)

        # Histogram the data
        hist_xy, x_edges, y_edges = np.histogram2d(
            xy[:, 0], xy[:, 1], bins=bins, range=(0, 1)
        )

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
        hist_x = np.histogram(xy[:, 0], bins=x_edges)[0]
        hist_y = np.histogram(xy[:, 1], bins=y_edges)[0]

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

    def counts(
        self,
        anode: PositionAnode,
        roi: ArrayLike = ((0, 1), (0, 1)),
        bkg: Spectrum | float | int | None = None,
    ) -> tuple[float, float]:
        """Get the number of counts in the spectrum and the estimated
        uncertainty.

        Parameters
        ----------
        anode: PositionAnode
            Anode instance with a `process` method that converts raw values
            timing values from the used anode to x, y coordinates.
        roi: array_like, shape (2,2), optional
            Region of interest for the detector in the form
            `((xmin, xmax), (ymin, ymax))`. If not provided, the
            full detector is used.
        bkg: Spectrum or float, optional
            Background spectrum (dark counts) to be subtracted that can
            be provided either as an instance of `Spectrum` or as a
            float. For this to work properly, the both spectra
            should be normalized to their measurement duration.

        Returns
        -------
        val: float
            The number of counts (normalized to measurement duration
            and any normalization parameters if available).
        err: float
            The propagated Poisson uncertainty.

        """
        # Coordinates of photon hits (x, y)
        xy = self.xy(anode)

        # Apply y roi filter
        xy = xy[xy[:, 1] > roi[1][0]]
        xy = xy[xy[:, 1] < roi[1][1]]

        # Discard y values
        xy = xy[:, 0].flatten()

        # Apply x roi filter
        xy = xy[xy > roi[0][0]]
        xy = xy[xy < roi[0][1]]

        # Calculate the number of counts and the Poisson uncertainty
        val = len(xy)
        err = np.sqrt(val)

        # Normalize data to measurement duration
        if self.time is not None:
            val /= self.time
            err /= self.time

        # Subtract background before further normalization
        if isinstance(bkg, Spectrum):
            if bkg.time is None or self.time is None:
                errmsg = "Can't subtract background without time information."
                raise ValueError(errmsg)

            bkg_val, bkg_err = bkg.counts(anode, roi=roi, bkg=None)
            # Using just the statistical uncertainty of the background
            # counts would underestimate the uncertainty of the subtraction
            bkg_err = np.sqrt(bkg_val * self.time) / self.time
            val = max(val - bkg_val, 0)
            err = np.sqrt(err**2 + bkg_err**2)

        elif isinstance(bkg, (int, float)):
            val = max(val - bkg, 0)

        elif bkg is not None:
            errmsg = "bkg must be Spectrum, int or float"
            raise TypeError(errmsg)

        # Normalize data to account for beam intensity, gas
        # pressure, etc.
        for normalization in self.norm:
            norm = self.norm[normalization]
            if isinstance(norm, float):
                val /= norm
                err /= norm

            else:
                norm_val = np.mean(norm)
                norm_err = np.std(norm, ddof=1, mean=norm_val)
                err = np.sqrt(
                    err**2 / norm_val**2 + norm_err**2 * val**2 / norm_val**4
                )
                val /= norm_val

        # Return the counts and the uncertainty
        return val, err

    def spectrum(
        self,
        anode: PositionAnode,
        bin_edges: ArrayLike,
        roi: ArrayLike = ((0, 1), (0, 1)),
        qeff: object | None = None,
        bkg: Spectrum | None = None,
        calib: object | None = None,
    ) -> tuple[NDArray, NDArray]:
        """Calculate the spectum and its uncertainties for a given
        set of bin edges.

        Parameters
        ----------
        anode: PositionAnode
            Anode instance with a `process` method that converts raw values
            timing values from the used anode to x, y coordinates.
        bin_edges: array_like
            Bin edges for the histogram. For a calibrated
            spectrum, bin edges should be in wavelength units.
            For an uncalibrated spectrum, these should be between
            0 and 1.
        roi: array_like, shape (2,2), optional
            Region of interest for the detector in the form
            `((xmin, xmax), (ymin, ymax))`. If not provided, assuming
            full detector with `((0, 1), (0, 1))`.
        qeff: [np.ndarray, np.ndarray, np.ndarray], optional
            Detector efficiencies in the form `(values, errors, x)`.
            The efficiencies are interpolated to 512 points between
            0 and 1.
        bkg: Spectrum, optional
            Background spectrum (dark counts) to be subtracted.
            For this to work properly, both spectra must be normalized
            to their measurement duration.
        calib: array_like, shape (2,2), optional
            Wavelength calibration parameters in the form
            `((a0, err), (a1, err))`, where `a0` and `a1`
            correspond to $\\lambda = a_1 x + a_0$ and `err` to the
            respective uncertainties.

        Returns
        -------
        spec: np.ndarray, shape (N,)
            The spectrum in the form of bin values.
        err: np.ndarray, shape (N,)
            The respective uncertainties of the bin values.

        """
        # Get x and y values of the photon hits
        data = self.xy(anode)

        # Parse the region of interest
        roi = np.array(roi)

        # Apply y roi filter
        y_min, y_max = roi[1]
        data = data[data[:, 1] > y_min]
        data = data[data[:, 1] < y_max]

        # Don't need y values anymore: project to x axis
        data = data[:, 0].flatten()

        # Apply calibration to data and roi
        if calib is not None:
            x_min, x_max = calib.calibrate(roi[0])
            det_bin_edges = calib.revert(bin_edges)
            data = calib.calibrate(data)

        else:
            x_min, x_max = roi[0]
            det_bin_edges = bin_edges

        # Histogram the data
        spec = np.histogram(data, bins=bin_edges)[0]
        spec = np.asarray(spec, dtype=np.float64)

        # Poisson uncertainties for each bin
        err = np.sqrt(spec)

        # Normalize to counts / s
        spec /= self.time
        err /= self.time

        if bkg is not None:
            bkg_spec, bkg_err = bkg.spectrum(
                anode, bin_edges, roi=roi, calib=calib, bkg=None, qeff=None
            )

            # Subtract background
            spec -= bkg_spec
            spec[spec < 0] = 0
            err = np.sqrt(err**2 + bkg_err**2)

        if qeff is not None:
            # Get the inverse efficiencies for the chosen binning
            inv_eff, inv_eff_err = qeff.inverse_efficiencies(det_bin_edges)

            # Correct for quantum efficiencies and propagate errors
            spec *= inv_eff
            err = np.sqrt(inv_eff**2 * err**2 + spec**2 * inv_eff_err**2)

        # Normalize data to account for beam intensity, gas
        # pressure, etc.
        for normalize in self._norm:
            norm_val, norm_err = getattr(self, normalize)
            err = np.sqrt(
                err**2 / norm_val**2 + norm_err**2 * spec**2 / norm_val**4
            )
            spec /= norm_val

        # Apply x roi filter
        idx_min = np.searchsorted(bin_edges, x_min)
        idx_max = np.searchsorted(bin_edges, x_max)
        spec[:idx_min] = 0
        spec[idx_max - 1 :] = 0
        err[:idx_min] = 0
        err[idx_max - 1 :] = 0

        # Return the spectrum and uncertainties
        return spec, err

    def transform_norm(self, norm: str, func: callable) -> None:
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
        if not hasattr(self, norm):
            errmsg = f"Unknown normalization {norm}"
            raise AttributeError(errmsg)

        # Get the current value and uncertainty
        val = getattr(self, norm)

        # Call function and propagate the uncertainty
        val, err = propagate(func, val[0], val[1] ** 2)

        # Set the transformed nomalization
        setattr(self, norm, np.array([val, np.sqrt(err)]))

    def convert_unit(self, norm: str, fro: str, to: str) -> None:
        """Convert the specified normalization values to a different
        unit using the pint package.

        Parameters
        ----------
        norm: str
            Name of the normalization parameter to convert.
        fro: str
            Unit to convert from (pint).
        to: str
            Unit to convert to (pint).

        """
        # Try to import pint
        try:
            from pint import UnitRegistry

        except ImportError as e:
            errmsg = "pint is required to convert units."
            raise ImportError(errmsg) from e

        ureg = UnitRegistry()

        # Convert the normalization values
        self.transform_norm(norm, lambda x: ureg.Quantity(x, fro).m_as(to))


def spectrum_from_h5(
    file: h5py.Group | str,
    scan: str = "0",
    step: str = "0.0",
    time: int | None = None,
    raw: str = "dld_rd#raw",
    **norm: str,
) -> Spectrum:
    """Load a Spectrum from an h5 file generated by metro2hdf.

    Parameters
    ----------
    h5: str or h5py.Group
        Open h5 file or path.
    scan: int, optional
        Scan index. In case of a single measurements / scan
        the index is `0`.
    step: str, optional
        Step value in a scan. If the measurement
        was not a scan, the step value is `0.0`.
    time: int, optional
        Measurement time in seconds for normalization.
    raw: str, optional
        Path to the raw data in the h5 file.
    **norm: str
        Path to additional normalization parameters as keyword
        arguments like the upstream intensity or target density.

    Returns
    -------
    spec: Spectrum
        Dataclass containing the loaded events.

    """
    if isinstance(file, str):
        if not os.path.exists(file):
            errmsg = "Could not find h5 file."
            raise ValueError(errmsg)

        if not file.endswith((".h5", ".hdf5")):
            errmsg = "Unknown file type."
            raise ValueError(errmsg)

        with h5py.File(file, "r") as h5:
            _extract_spectrum_from_h5(
                h5,
                scan=scan,
                step=step,
                raw=raw,
                time=time,
                **norm,
            )

    elif isinstance(file, h5py.Group):
        _extract_spectrum_from_h5(
            file,
            scan=scan,
            step=step,
            raw=raw,
            time=time,
            **norm,
        )

    else:
        errmsg = "file must be open h5 file or the path."
        raise TypeError(errmsg)


def _extract_spectrum_from_h5(
    h5: h5py.Group,
    scan: str = "0",
    step: str = "0.0",
    raw: str = "dld_rd#raw",
    time: int | None = None,
    **norm: str,
) -> Spectrum:
    # Appends scan index to the path
    group_raw = raw + "/" + scan

    # Check if the data is found
    if group_raw not in h5:
        errmsg = f"{group_raw} not found."
        raise KeyError(errmsg)

    if step not in h5[group_raw]:
        errmsg = f"{step} not found in {group_raw}"
        raise KeyError(errmsg)

    # Load the raw data
    group_raw = h5[group_raw]
    data_raw = np.array(group_raw[step])

    # Load normalization values
    for name, group in norm.items():
        group_norm = group + "/" + scan

        if group_norm not in h5:
            errmsg = f"{group_norm} not found."
            raise KeyError(errmsg)

        # Load the dataset / group
        data_norm = h5[group_norm]

        # Pass this to Spectrum
        spec_norm = {}

        # Handle different types of recorded data
        if isinstance(data_norm, h5py.Dataset):
            # Values averaged by metro
            data_norm = np.asarray(data_norm)

            if data_norm.shape == (1,):
                spec_norm[name] = float(data_norm[0])

            elif data_norm.ndim == 1:
                idx = list(group_raw.keys()).index(step)
                spec_norm[name] = np.array([data_norm[idx]])

            else:
                errmsg = f"Could not parse data in {group_norm}."
                raise RuntimeError(errmsg)

        elif step in data_norm:
            # Values recorded by metro every x seconds
            spec_norm[name] = np.array(data_norm[step])

        else:
            errmsg = f"{step} not found in {group_norm}."
            raise KeyError(errmsg)

    # Initialize the Spectrum dataclass
    return Spectrum(data_raw, time=time, norm=spec_norm, step=step, scan=scan)


@njit()
def compute_bin(x: float, bin_edges: NDArray) -> int:
    # assuming uniform bins
    n = bin_edges.shape[0] - 1
    a_min = bin_edges[0]
    a_max = bin_edges[-1]

    # special case to mirror NumPy behavior for last bin
    if x == a_max:
        return n - 1  # a_max always in last bin

    bin_idx = int(n * (x - a_min) / (a_max - a_min))

    if bin_idx < 0 or bin_idx >= n:
        return None

    else:
        return bin_idx


@njit()
def numba_histogram(data: NDArray, bin_edges: NDArray) -> NDArray:
    hist = np.zeros((bin_edges.shape[0] - 1,), dtype=np.float64)

    for x in data.flat:
        bin_idx = compute_bin(x, bin_edges)
        if bin_idx is not None:
            hist[int(bin_idx)] += 1

    return hist


@njit()
def numba_weighted_histogram(
    data: NDArray,
    bin_edges: NDArray,
    weights: NDArray,
) -> NDArray:
    hist = np.zeros((bin_edges.shape[0] - 1,), dtype=np.float64)

    for x, w in zip(data.flat, weights.flat):
        bin_idx = compute_bin(x, bin_edges)
        if bin_idx is not None:
            hist[int(bin_idx)] += w

    return hist


@njit(parallel=True)
def montecarlo_spectrum(
    spectrum: NDArray,
    data: NDArray,
    xedges: NDArray,
    rng: np.random.Generator,
    mc_samples: int,
    calib: NDArray,
    bkg: tuple[NDArray, float] | None,
    qeff: tuple[NDArray, NDArray, NDArray] | None,
) -> NDArray:
    # Prepare data
    data_counts = data.shape[0]

    # Define the interpolation grid
    xe = np.linspace(0, 1, 513)
    x = (xe[1:] + xe[:-1]) * 0.5

    # Assign the data points to the bins
    inds = np.digitize(data, xe[1:-1])

    # Prepare the quantum efficiency correction
    interp_qeff = False
    data_eff = np.ones(data_counts, dtype=np.float64)

    if qeff is not None:
        qeff_val, qeff_err, qeff_x = qeff

        interp_qeff = True

        n_eff = qeff_val.shape[0]

    # Prepare the background subtraction
    subtr_bkg = False
    if bkg is not None:
        bkg_data, bkg_ratio = bkg

        subtr_bkg = True

        bkg_sample_size = int(bkg_data.shape[0] * bkg_ratio)

        # Calculate the background distribution
        bkg_pdf = numba_histogram(bkg_data, xe)

        # Assign the background probabilities to the data points
        bkg_prob = bkg_pdf[inds]

    # Start the Monte Carlo simulation
    for i in prange(mc_samples):
        # Create a sample of the efficiencies
        if interp_qeff:
            eff_sample = np.ones(n_eff, dtype=np.float64)
            for j in range(n_eff):
                eff_sample[j] = rng.normal(qeff_val[j], qeff_err[j], size=1)[0]

            # Interpolate the efficiencies to get a smoother spectrum
            eff_sample = np.interp(x, qeff_x, eff_sample)

            # Get the efficiencies for each point
            data_eff = 1 / eff_sample[inds]

        # Select data points based on Poisson sampling
        p = rng.poisson(lam=data_counts, size=1)[0]
        poisson_inds = rng.integers(0, data_counts, size=p)
        data_sample = data[poisson_inds]
        data_eff_sample = data_eff[poisson_inds]

        if subtr_bkg:
            bkg_sample = bkg_prob[poisson_inds]

            # Remove data points based on the background distribution
            p = rng.poisson(lam=bkg_sample_size, size=1)[0]
            bkg_cdf = np.cumsum(bkg_sample)
            remove_inds = np.searchsorted(bkg_cdf, rng.random(p) * bkg_cdf[-1])
            data_sample = np.delete(data_sample, remove_inds)
            data_eff_sample = np.delete(data_eff, remove_inds)

        # Convert x values to wavelengths
        a0_sample = rng.normal(calib[0, 0], calib[0, 1], size=1)[0]
        a1_sample = rng.normal(calib[1, 0], calib[1, 1], size=1)[0]
        data_sample = a1_sample * data_sample + a0_sample

        # Calculate the sum of weights for each bin, i.e. the weighted spectrum
        spectrum[i] = numba_weighted_histogram(
            data_sample, xedges, data_eff_sample
        )

    # Return the n generated Monte Carlo spectra
    return spectrum
