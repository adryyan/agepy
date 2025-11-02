"""Load, manage and evaluate fluorescence spectra of a PhexPhem map."""

from __future__ import annotations

from pathlib import Path
import yaml
import numpy as np
import pandas as pd

from agepy.spec.data import MetroValue
from agepy.spec.utils import open_metro_h5, load_metro_scan
from .spectrum import Spectrum

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray

    roi_tuple = tuple[tuple[float, float], tuple[float, float]]


class PhexPhem:
    mapping: pd.DataFrame
    spectra: dict = {}
    target_density: dict = {}
    photon_flux: dict = {}
    bkg: dict = {}
    ref: dict = {}
    qeff: dict = {}
    calib: dict = {}

    def __init__(self, data: str) -> None:
        # Path to the directory containing beamtime directories
        data_dir = Path(data)

        if not data_dir.is_dir():
            errmsg = f"{data} is not a directory"
            raise ValueError(errmsg)

        # Recursively look for the scans.csv file
        pattern = "scans.csv"
        matches = list(data_dir.rglob(pattern))

        if len(matches) == 0:
            errmsg = "Could't find any scans.csv"
            raise FileNotFoundError(errmsg)

        # Paths to the data directories
        self.metro = {}

        #
        col_dtype = {"num": str}
        scans = {}

        for match in matches:
            # Beamtime directory
            beamtime_dir = match.resolve().parent
            beamtime_name = beamtime_dir.name

            # Load the metro settings relevant to reading the h5 files
            config = beamtime_dir / "metro.yaml"

            if not config.is_file():
                errmsg = f"Couldn't find metro.yaml for {beamtime_name}"
                raise FileNotFoundError(errmsg)

            with open(config, "r") as f:
                self.metro[beamtime_name] = yaml.safe_load(f)

            #
            h5_path = self.metro[beamtime_name]["photon_energy"]["data"] + "/0"

            # Path to the measurement data
            beamtime_data = beamtime_dir / "data"
            self.metro[beamtime_name]["path"] = beamtime_data

            # Load the scan info from scans.csv
            df = pd.read_csv(match.resolve(), index_col="num", dtype=col_dtype)

            measurements = {}

            for num, row in df.iterrows():
                # Get the steps in the scan
                with open_metro_h5(num, data_dir=beamtime_data) as h5:
                    steps = list(h5[h5_path].keys())
                    energies = [float(dset) for dset in h5[h5_path].values()]

                measurements[num] = pd.concat(
                    [row.to_frame().T] * len(steps), ignore_index=True
                )

                measurements[num].index = steps
                measurements[num]["phex_energy"] = energies

            measurements = pd.concat(measurements)

            scans[beamtime_name] = measurements

        self.m_map = pd.concat(scans)
