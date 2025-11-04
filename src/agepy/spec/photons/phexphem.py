"""Load, manage and evaluate fluorescence spectra of a PhexPhem map."""

from __future__ import annotations

from pathlib import Path
import yaml
import numpy as np
import pandas as pd

from agepy.spec.metro import open_metro_h5, load_data_stream, read_steps
from .spectrum import Spectrum

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray


class PhexPhem:
    def __init__(self, data_dir: str) -> None:
        # Path to the directory containing beamtime directories
        data_dir = Path(data_dir)

        # Recursively look for the scans.csv file
        matches = list(data_dir.rglob("scans.csv"))

        if len(matches) == 0:
            errmsg = "Could't find any scans.csv"
            raise FileNotFoundError(errmsg)

        # Metadata specific to beamtimes
        self.beamtimes = {}

        # Look up of loaded data
        self.data = {}
        self.ref = {}
        self.qeff = {}
        self.calib = {}

        # DataFrame mapping steps to related data
        step_map = {}

        # Build the mapping DataFrame from config files found in the
        # glob'ed directories

        # config: scans.csv
        col_dtype = {
            "num": str,  # metro measurement number, e.g. "042"
            "time": int,  # time per step in the scan
            "grating_pos": int,  # grating position in Å (center wavelength)
            "ref": str,  # corresponding reference measurement (num)
            "qeff": str,  # corresponding quantum efficiency measurement (num)
        }

        # Loop over the found beamtimes
        for match in matches:
            # Use .resolve() to ensure a parent exists
            match = match.resolve()

            # Get the path of the beamtime directory
            beamtime_dir = match.parent

            # Use the directory name as the identifier for the beamtime
            beamtime = beamtime_dir.name

            # Load the metro settings relevant to reading the h5 files
            config = beamtime_dir / "metro.yaml"

            if not config.is_file():
                errmsg = f"Couldn't find metro.yaml for {beamtime}"
                raise FileNotFoundError(errmsg)

            with open(config, "r") as f:
                self.beamtimes[beamtime] = yaml.safe_load(f)

            # The measurement data is expected to be in hdf5 files
            # produced by metro2hdf in the subdirectory data
            beamtime_data = beamtime_dir / "data"
            self.beamtimes[beamtime]["data"] = beamtime_data

            # Get the path to the group in the hdf5 data files
            # containing the measured beamline energies (step values)
            group = self.beamtimes[beamtime]["beamline_energy"]["data"]

            # Load the scan info from scans.csv
            df = pd.read_csv(
                match,
                index_col="num",
                usecols=col_dtype.keys(),
                dtype=col_dtype,
            )

            # Load the measured steps for each scan from the data files
            # and add them to the DataFrame
            measurements = {}

            for num, row in df.iterrows():
                with open_metro_h5(num, data_dir=beamtime_data) as h5f:
                    # Dataset names (step values) for lazy loading
                    steps = read_steps(h5f)

                    # Load the measured beamline energies
                    energies = np.array(
                        load_data_stream(h5f, group), dtype=np.float64
                    ).flatten()

                # Create a new entry for each step
                # TODO: Test if this is too slow
                measurements[num] = pd.concat(
                    [row.to_frame().T] * len(steps), ignore_index=True
                )

                # Set the dataset names as the index and add the energies
                measurements[num].index = steps
                measurements[num]["beamline_energy"] = energies

            # Create a DataFrame with num as a MultiIndex
            step_map[beamtime] = pd.concat(measurements)

        # Create a DataFrame with beamtime as a MultiIndex
        self.step_map = pd.concat(step_map)
