"""Load, manage and evaluate fluorescence spectra of a PhexPhem map."""

from __future__ import annotations

from pathlib import Path
import yaml
import numpy as np
import pandas as pd

from agepy.spec.metro import open_metro_h5, load_data_stream, read_steps

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray


class PhexPhem:
    def __init__(self, data_dir: str, verbose: int = 0) -> None:
        # Path to the directory containing beamtime directories
        data_dir = Path(data_dir)

        # Recursively look for the phexphem file
        matches = list(data_dir.rglob("phexphem.yaml"))

        if len(matches) == 0:
            errmsg = "Could't find any phexphem.yaml"
            raise FileNotFoundError(errmsg)

        # Config specific to beamtimes
        self.beamtimes = {}

        # Build DataFrames from config files found in the
        # glob'ed directories
        spectra = {}
        ref = {}
        qeff = {}

        # config: scans.csv
        scans_dtype = {
            "num": str,  # metro measurement number, e.g. "042"
            "time": int,  # time per step in the scan
            "grating_pos": int,  # grating position in Å (center wavelength)
            "ref": str,  # corresponding reference measurement (num)
            "qeff": str,  # corresponding quantum efficiency measurement (num)
        }

        # config: ref.csv
        ref_dtype = {
            "num": str,  # metro measurement number, e.g. "042"
            "time": int,  # measurement time
            "target": str,  # target atom / molecule, e.g. "H2"
            "beamline_energy": float,  # requested beamline energy
            "grating_pos": int,  # grating position in Å (center wavelength)
            "slit": int,  # beamline exit slit in μm
            "det_voltage": str,  # detector (mcp) voltage in V
            "comment": str,
        }

        # config: qeff.csv
        ref_dtype = {
            "num": str,  # metro measurement number, e.g. "042"
            "time": int,  # time per step in the scan
            "target": str,  # target atom / molecule, e.g. "H2"
            "beamline_energy": float,  # requested beamline energy
            "comment": str,  # the expected emission line
        }

        # Loop over the found beamtimes
        for match in matches:
            # Use .resolve() to ensure a parent exists
            match = match.resolve()

            # Get the path of the beamtime directory
            beamtime_dir = match.parent

            # Use the directory name as the identifier for the beamtime
            beamtime = beamtime_dir.name

            if verbose > 0:
                print(f"Parsing beamtime {beamtime}...")

            # Load the phexphem measurement settings
            with open(match, "r") as f:
                self.beamtimes[beamtime] = yaml.safe_load(f)

            # The measurement data is expected to be in hdf5 files
            # produced by metro2hdf in the subdirectory data
            beamtime_data = beamtime_dir / "data"
            self.beamtimes[beamtime]["data"] = beamtime_data

            # Get the path to the group in the hdf5 data files
            # containing the measured beamline energies (step values)
            group = self.beamtimes[beamtime]["beamline_energy"]["data"]

            # Get the path to the scans.csv
            scans_csv = beamtime_dir / "scans.csv"

            if not scans_csv.is_file():
                errmsg = f"Couldn't find scans.csv for {beamtime}"
                raise FileNotFoundError(errmsg)

            # Load the scan info from scans.csv
            df = pd.read_csv(
                scans_csv,
                index_col="num",
                usecols=scans_dtype.keys(),
                dtype=scans_dtype,
            )

            # Load the measured steps for each scan from the data files
            # and add them to the DataFrame
            measurements = {}

            for num, row in df.iterrows():
                if verbose > 1:
                    print(f"  Parsing measurement {num}...")

                with open_metro_h5(num, data_dir=beamtime_data) as h5f:
                    # Dataset names (step values) for lazy loading
                    steps = read_steps(h5f)

                    # Load the measured beamline energies
                    energies = np.asarray(load_data_stream(h5f, group))

                n = energies.size
                m = {col: np.full(n, val) for col, val in row.items()}
                m["beamline_energy"] = energies
                measurements[num] = pd.DataFrame(m)

                # Set the dataset names as the index
                measurements[num].index = steps

            # Create a DataFrame with num as a MultiIndex
            spectra[beamtime] = pd.concat(measurements)

            # Get the path to the ref.csv
            ref_csv = beamtime_dir / "ref.csv"

            if not ref_csv.is_file():
                errmsg = f"Couldn't find ref.csv for {beamtime}"
                raise FileNotFoundError(errmsg)

            # Load the scan info from scans.csv
            ref[beamtime] = pd.read_csv(
                ref_csv,
                index_col="num",
                usecols=ref_dtype.keys(),
                dtype=ref_dtype,
            )

            # Get the path to the ref.csv
            qeff_csv = beamtime_dir / "qeff.csv"

            if not qeff_csv.is_file():
                errmsg = f"Couldn't find qeff.csv for {beamtime}"
                raise FileNotFoundError(errmsg)

            # Load the scan info from scans.csv
            qeff[beamtime] = pd.read_csv(
                qeff_csv,
                index_col="num",
                usecols=ref_dtype.keys(),
                dtype=ref_dtype,
            )

        # Create a DataFrames with beamtime as a MultiIndex
        self.spectra = pd.concat(spectra)
        self.ref = pd.concat(ref)
        self.qeff = pd.concat(qeff)

        # Create columns for the data
        self.spectra["spectrum"] = None
        self.spectra["target_density"] = None
        self.spectra["beamline_flux"] = None
        self.ref["spectrum"] = None
        self.ref["target_density"] = None
        self.ref["beamline_flux"] = None
        self.qeff["spectrum"] = None
        self.qeff["target_density"] = None
        self.qeff["beamline_flux"] = None
