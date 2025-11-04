from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import h5py
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray


@contextmanager
def open_metro_h5(measurement: str, data_dir: str = ".") -> h5py.File:
    """Open an hdf5 file produced by metro2hdf.

    Convenience wrapper around h5py.File for easier access using
    the measurement number to glob the hdf5 file in the specified
    directory.

    Parameters
    ----------
    measurement: str
        Metro measurement number (e.g. `"042"`).
    data_dir: str, optional
        Path to the directory containing the hdf5 file.

    Yields
    ------
    h5py.File
        Open hdf5 file.

    """
    # glob pattern
    pattern = f"{measurement}*.h5"

    # Get matching file path
    match = list(Path(data_dir).glob(pattern))

    if len(match) == 0:
        errmsg = f"Could not find measurement {measurement}"
        raise FileNotFoundError(errmsg)

    with h5py.File(match[0].resolve(), "r") as h5f:
        yield h5f


def load_data_stream(
    h5f: h5py.File,
    data: str,
    scan_idx: str = "0",
    step_idx: str | None = None,
) -> list[NDArray] | NDArray:
    """Load 'continuous' metro data streams from a scan
    in an open hdf5 file.

    If a step index `step_idx` is specified, the data is
    returned as a single `np.ndarray`.

    Parameters
    ----------
    h5f: h5py.File
        Open hdf5 file (see `open_metro_h5`).
    data: str
        Name of the metro data stream (e.g. `"device#value"`).
    scan_idx: str, optional
        Index of the scan to be loaded. Usually `"0"` in case of
        measurements with one or no scan.
    step_idx: str, optional
        Index of the step to be loaded. If `None` all datasets
        in the scan are returned.

    Returns
    -------
    data: list of np.ndarray or np.ndarray
        Dataset(s).

    """
    # Check if the data stream is found
    if data not in h5f:
        errmsg = f"Data {data} not found"
        raise KeyError(errmsg)

    # Check if the data is a continuous data stream (metro)
    if (
        "Frequency" not in data.attrs
        or h5f[data].attrs["Frequency"] != "continuous"
    ):
        errmsg = f"Data {data} is not 'continuous'"
        raise ValueError(errmsg)

    # Append the scan index to the path
    scan = data + "/" + scan_idx

    # Check if the scan index is present
    if scan not in h5f:
        errmsg = f"Scan index {scan_idx} not found in {data}"
        raise KeyError(errmsg)

    if step_idx is None:
        return [np.array(dset) for dset in h5f[scan].values()]

    # Append the step index to the path
    step = scan + "/" + step_idx

    # Check if the scan index is present
    if step not in h5f:
        errmsg = f"Scan index {step_idx} not found in {scan}"
        raise KeyError(errmsg)

    return np.array(h5f[step])


def read_steps(
    h5f: h5py.File,
    scan_idx: str = "0",
) -> list[str]:
    """Read the names of the datasets in a metro scan (step values).

    Parameters
    ----------
    h5f: h5py.File
        Open hdf5 file (see `open_metro_h5`).
    scan_idx: str, optional
        Index of the scan to be loaded. Usually `"0"` in case of
        measurements with one or no scan.

    Returns
    -------
    list of str
        Dataset names in the scan.

    """
    for data in h5f:
        # Check for the Frequency attribute
        if "Frequency" not in data.attrs:
            continue

        # Check data contains continuous data streams
        if h5f[data].attrs["Frequency"] == "continuous":
            # Append the scan index to the path
            scan = data + "/" + scan_idx

            # Check if the scan index is present
            if scan not in h5f:
                errmsg = f"Scan index {scan_idx} not found in {data}"
                raise KeyError(errmsg)

            return list(h5f[scan].keys())

    errmsg = f"Could not find steps in {h5f.filename}"
    raise ValueError(errmsg)
