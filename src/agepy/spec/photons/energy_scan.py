"""Load and process exciting-photon energy scans of H2."""

from __future__ import annotations

import warnings
import pickle
import numpy as np
from scipy.stats import norm
from jacobi import propagate
import pandas as pd
import matplotlib.pyplot as plt

from .scan import Scan

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal, Sequence
    from numpy.typing import NDArray, ArrayLike
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from .anodes import PositionAnode
    from .spectrum import Spectrum


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
        col = ["J", "Elp", "vp", "Jp", "fit", "exc_energy"]
        self._phex = pd.DataFrame(columns=col)

        col = ["vpp", "Jpp", "fit"]
        self._phem = pd.DataFrame(columns=col)
        self._phem.index = pd.MultiIndex.from_tuples(
            [], names=["phex", "phem"]
        )

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

        # Remove the uncertainties
        self._energy_uncertainty = np.delete(self._energy_uncertainty, inds)

    def select_by_phex(
        self,
        phex: dict[str, str | int],
        n_std: int = 1,
        ignore_overlap: bool = False,
    ) -> tuple[int, NDArray]:
        # Find the phex assignment
        df = self._phex_assignments.copy()
        for key, value in phex.items():  # noqa B007
            df.query(f"{key} == @value", inplace=True)

            # Check if there are matching assignments
            if df.empty:
                raise ValueError("Phex assignment not found.")

        # At this point there should be only one assignment
        assert len(df) == 1, "Multiple or no assignments found."

        # Get the fit results
        fit_val = df["val"].iloc[0]

        # Select energy steps within n_std standard deviations of the mean
        step_idx = np.argwhere(
            np.abs(self.steps - fit_val[1]) < fit_val[2] * n_std
        ).flatten()

        # Check if steps were found
        if len(step_idx) == 0 or ignore_overlap:
            return df.index[0], step_idx

        # Define energy range
        e_range = (  # noqa F841
            fit_val[1] - fit_val[2] * n_std,
            fit_val[1] + fit_val[2] * n_std,
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
                wrnmsg = "No steps found without overlap"
                warnings.warn(wrnmsg, stacklevel=1)

            else:
                step_idx = overlap_idx

        return df.index[0], step_idx

    def assign_phex(
        self,
        reference: pd.DataFrame,
        energy_range: float = 0.05,
    ) -> int:
        """Interactively assign excitations to peaks in the
        exciting-photon energy spectrum.

        Parameters
        ----------
        reference: pd.DataFrame
            DataFrame containing excitation energies in eV with the
            corresponding transition labels: J, Elp, vp, Jp.
        energy_range: float
            Energy range length to show in one plot.

        """
        from agepy.interactive import get_qapp
        from ._interactive_phex import AssignPhex

        # Get the Qt application
        app = get_qapp()

        # Intialize the viewer
        mw = AssignPhex(self, reference, energy_range)
        mw.show()

        # Run the application
        return app.exec()

    def save_phex(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self._phex_assignments, f)

    def load_phex(self, path: str) -> None:
        with open(path, "rb") as f:
            self._phex_assignments = pickle.load(f)

    def eval_phex(
        self,
        reference: pd.DataFrame,
        plot: bool = True,
        plot_pulls: bool = True,
    ) -> tuple[Figure, Axes]:
        # Create a list of fit values and reference values
        fit_energies = []
        ref_energies = []
        labels = []

        # Find matching transitions
        for row in self._phex_assignments.itertuples():
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

        except ImportError as e:
            errmsg = "iminuit is required for fitting"
            raise ImportError(errmsg) from e

        # Define the cost function
        c = LeastSquares(
            ref_energies, fit_energies[:, 0], fit_energies[:, 1], model
        )

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
                "Bad assignment of:",
                labels[idx],
                "with diff:",
                c.prediction(m.values)[idx] - fit_energies[idx, 0],
            )

        # Plot the results
        if plot:
            # Create the figure
            fig, ax = plt.subplots(
                2, 1, sharex=True, gridspec_kw={"hspace": 0.3}
            )

            # Plot the assignments
            ax[0].errorbar(
                ref_energies,
                fit_energies[:, 0],
                yerr=fit_energies[:, 1],
                fmt="s",
                markersize=1.5,
                label="Assign. Phex",
            )

            # Plot the fit results
            ax[0].plot(
                ref_energies, c.prediction(m.values), label="Lin. Regr."
            )

            # Plot the legend with the chi2/ndof
            chi2ndof = m.fmin.reduced_chi2
            ax[0].legend(
                title=r"$\chi^2\;/\;$ndof = " + f"{chi2ndof:.2f}", loc="best"
            )

            # Set title and labels
            ax[0].set_title(
                r"Assigned Photon-Excitation Energies $E_\text{data}$"
            )
            ax[0].set_ylabel(r"$E_\text{data}$ [eV]")
            ax[1].set_xlabel(r"$E_\text{literature}$ [eV]")

            # Plot the residuals
            ax[1].axhline(0, color="black", linestyle="--", alpha=0.9)
            if plot_pulls:
                # Plot the pulls
                ax[1].step(ref_energies, c.pulls(m.values), where="mid")

                # Set title and label
                ax[1].set_title(
                    "Studentized Residuals of the Linear Regression"
                )
                ax[1].set_ylabel("Pulls")

            else:
                # Plot the differences of the fit to the data
                ax[1].step(
                    ref_energies,
                    c.prediction(m.values) - fit_energies[:, 0],
                    where="mid",
                )

                # Set title and label
                ax[1].set_title("Difference to the Linear Regression")
                ax[1].set_ylabel(r"$(\text{Lin. Regr.} - E_\text{data})$ [eV]")

            return fig, ax

        return None, None

    def assign_phem(
        self,
        reference: pd.DataFrame,
        calib: tuple[float, float],
        bins: int = 512,
    ) -> int:
        """Interactively assign transitions to peaks in the
        fluorescence spectra.

        Parameters
        ----------
        reference: pd.DataFrame
            DataFrame containing emission energies in nm with the
            corresponding transition labels: Elp, vp, Jp, vpp, Jpp.
        calib: [float, float]
            Initial guess of the calibration parameters (a0, a1).
        bins: int or array_like
            Bin numbers or edges.

        """
        from agepy.interactive import get_qapp
        from ._interactive_phem import AssignPhem

        # Get the Qt application
        app = get_qapp()

        # Intialize the viewer
        mw = AssignPhem(self, reference, calib, bins)
        mw.show()

        # Run the application
        return app.exec()

    def save_phem(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self._phem, f)

    def load_phem(self, path: str) -> None:
        with open(path, "rb") as f:
            self._phem = pickle.load(f)

    def eval_phem(
        self,
        reference: pd.DataFrame,
        calib_guess: tuple[float, float],
        plot: bool = True,
        plot_pulls: bool = True,
    ) -> tuple[Figure, Axes]:
        # Create a list of fit values and reference values
        fit_wl = []
        ref_wl = []
        labels = []

        # Find matching transitions
        for row in self._phem_assignments.itertuples():
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

        except ImportError as e:
            errmsg = "iminuit is required for fitting"
            raise ImportError(errmsg) from e

        # Define the cost function
        c = LeastSquares(ref_wl, fit_wl[:, 0], fit_wl[:, 1], model)

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
                "Bad assignment of:",
                labels[idx],
                "with diff:",
                c.prediction(m.values)[idx] - fit_wl[idx, 0],
            )

        # Plot the results
        if plot:
            # Create the figure
            fig, ax = plt.subplots(
                2, 1, sharex=True, gridspec_kw={"hspace": 0.3}
            )

            # Plot the assignments
            ax[0].errorbar(
                ref_wl,
                fit_wl[:, 0],
                yerr=fit_wl[:, 1],
                fmt="s",
                markersize=1.5,
                label="Assign. Phem",
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
                ax[1].set_title(
                    "Studentized Residuals of the Linear Regression"
                )
                ax[1].set_ylabel("Pulls")
            else:
                # Plot the differences of the fit to the data
                ax[1].step(
                    ref_wl, c.prediction(m.values) - fit_wl[:, 0], where="mid"
                )

                # Set title and label
                ax[1].set_title("Difference to the Linear Regression")
                ax[1].set_ylabel(
                    r"$(\text{Lin. Regr.} - x_\text{data})$ [arb. u.]"
                )

            return fig, ax

        return None, None

    def save_calib(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.calib, f)

    def load_calib(self, path: str) -> None:
        with open(path, "rb") as f:
            self.calib = pickle.load(f)

    def assigned_spectrum(
        self,
        J: int,
        Elp: str,
        vp: int,
        Jp: int,
        n_std: int | float = 1,
        normalize: bool = False,
        bins: int | ArrayLike = 512,
        qeff: bool = True,
        bkg: bool = True,
        calib: bool = True,
        uncertainties: Literal["montecarlo", "poisson"] = "montecarlo",
        mc_samples: int = 10000,
        mc_seed: int | None = None,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Get the spectrum from an assigned excitation.

        Parameters
        ----------
        step: int or float or str
            Step value to get the spectrum for. The value is
            converted to float and the closest step value
            is used.
        bins: int or array_like, optional
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
        # Find the phex assignment
        phex_idx, step_idx = self.select_by_phex(J, Elp, vp, Jp, n_std=n_std)

        if len(step_idx) == 0:
            wrnmsg = "Trying with 2 * n_std."
            warnings.warn(wrnmsg, stacklevel=1)
            phex_idx, step_idx = self.select_by_phex(phex, 2 * n_std)

        if len(step_idx) == 0:
            wrnmsg = "Ignore overlapping assignments."
            warnings.warn(wrnmsg, stacklevel=1)
            phex_idx, step_idx = self.select_by_phex(
                phex, n_std, ignore_overlap=True
            )

        if len(step_idx) == 0:
            wrnmsg = "No steps found for the given photon excitation."
            warnings.warn(wrnmsg, stacklevel=1)
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
                idx,
                edges,
                roi=roi,
                qeff=qeff,
                bkg=bkg,
                calib=calib,
                err_prop=err_prop,
                mc_samples=mc_samples,
            )

            # Normalize with the excitation fit results
            if normalize:
                # Define function for evaluating the excitation
                def calc_exc(par):
                    y = norm.pdf(self.steps[idx], par[0], par[1])  # noqa: B023
                    y_max = norm.pdf(par[0], par[0], par[1])
                    return y / y_max

                # Evaluate the excitation at the step value
                exc, exc_err = propagate(
                    calc_exc, fit_val[1:], fit_err[1:] ** 2
                )
                exc_err = np.sqrt(exc_err)

                # Normalize the spectrum
                spec /= exc
                err = np.sqrt(err**2 / exc**2 + spec**2 * exc_err**2 / exc**4)

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

    def phexphem(
        self,
        xedges: NDArray = None,
        yedges: NDArray = None,
        roi: tuple[tuple[float, float], tuple[float, float]] = None,
        qeff: bool = True,
        bkg: bool = True,
        calib: bool = True,
    ) -> NDArray:
        """ """
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
