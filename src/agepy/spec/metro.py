from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import h5py
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MetroLoader:
    def __init__(self, data_dir: str | Path = "."):
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)

        # Look for data files
        matches = list(data_dir.glob("[0-9][0-9][0-9]_*.h5"))

        if len(matches) == 0:
            errmsg = "Could't find any hdf5 data files"
            raise FileNotFoundError(errmsg)

        self.measurements = {}

        for match in matches:
            num = match.name[:3]
            self.measurements[num] = metroload(num, data_dir)

    def __contains__(self, num: str) -> callable:
        return num in self.measurements

    def __getitem__(self, num: str) -> callable:
        return self.measurements[num]


def metroload(
    num: str, data_dir: str | Path = "."
) -> callable[[str, str, str | None], dict[str, NDArray] | NDArray]:
    """Load data from an hdf5 file created by metro2hdf.

    Parameters
    ----------
    num: str
        Metro measurement number (e.g. `"042"`).
    data_dir: str or Path, optional
        Path to the directory containing the hdf5 file.

    Returns
    -------
    callable
        Loader function.

    Examples
    --------
    >>> data = metroload("042", data_dir="2024-07-BESSY-H2/")
    >>> spec = data("dld_rd#raw", step_key="12.269")

    """
    # Cache once loaded data for faster return next time
    cache = {}

    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)

    # Get matching file path
    match = list(data_dir.glob(f"{num}*.h5"))

    if len(match) == 0:
        errmsg = f"Could not find measurement {num}"
        raise FileNotFoundError(errmsg)

    data_file = match[0].resolve()

    def loader(
        data_key: str, scan_key: str = "0", step_key: str | None = None
    ) -> dict[str, NDArray] | NDArray:
        """Loads specified data from the hdf5 file.

        When one or all steps for one data key are loaded all steps
        are cached for future calls.

        Parameters
        ----------
        data_key: str
            Name of the metro data stream (e.g. `"device#value"`).
        scan_key: str, optional
            Index of the scan to be loaded. Usually `"0"` in case of
            measurements with one or no scan.
        step_key: str, optional
            Index of the step to be loaded. If `None` all datasets
            in the scan are returned.

        Returns
        -------
        dict or np.ndarray
            Dictionary of names (step values) and corresponding
            datasets or a single dataset if `step_key` is specified.

        """
        if data_key in cache and scan_key in cache[data_key]:
            if step_key is None:
                return cache[data_key][scan_key].copy()

            elif step_key in cache[data_key][scan_key]:
                return cache[data_key][scan_key][step_key]

            else:
                errmsg = f"Step {step_key} not found in {data_key}/{scan_key}"
                raise KeyError(errmsg)

        with h5py.File(data_file, "r") as h5f:
            data = load_data_stream(
                h5f, data_key, scan_key=scan_key, step_key=None
            )

        # Update the cache
        cache.update({data_key: {scan_key: data}})

        return loader(data_key, scan_key=scan_key, step_key=step_key)

    return loader


@contextmanager
def open_metro_h5(num: str, data_dir: str | Path = ".") -> h5py.File:
    """Open an hdf5 file produced by metro2hdf.

    Convenience wrapper around h5py.File for easier access using
    the measurement number to glob the hdf5 file in the specified
    directory.

    Parameters
    ----------
    num: str
        Metro measurement number (e.g. `"042"`).
    data_dir: str, optional
        Path to the directory containing the hdf5 file.

    Yields
    ------
    h5py.File
        Open hdf5 file.

    """
    # Create a Path instance
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)

    # glob pattern
    pattern = f"{num}*.h5"

    # Get matching file path
    match = list(data_dir.glob(pattern))

    if len(match) == 0:
        errmsg = f"Could not find measurement {num}"
        raise FileNotFoundError(errmsg)

    with h5py.File(match[0].resolve(), "r") as h5f:
        yield h5f


def load_data_stream(
    h5f: h5py.File,
    data_key: str,
    scan_key: str = "0",
    step_key: str | None = None,
) -> dict[str, NDArray] | NDArray:
    """Load 'continuous' metro data streams from a scan
    in an open hdf5 file.

    If a step index `step_key` is specified, the data is
    returned as a single `np.ndarray`.

    Parameters
    ----------
    h5f: h5py.File
        Open hdf5 file (see `open_metro_h5`).
    data_key: str
        Name of the metro data stream (e.g. `"device#value"`).
    scan_key: str, optional
        Index of the scan to be loaded. Usually `"0"` in case of
        measurements with one or no scan.
    step_key: str, optional
        Index of the step to be loaded. If `None` all datasets
        in the scan are returned.

    Returns
    -------
    data: dict or np.ndarray
        Dictionary of names (step values) and corresponding
        datasets or a single dataset if `step_key` is specified.

    """
    # Check if the data stream is found
    if data_key not in h5f:
        errmsg = f"Data {data_key} not found"
        raise KeyError(errmsg)

    # Check if the data is a continuous data stream (metro)
    if (
        "Frequency" not in h5f[data_key].attrs
        or h5f[data_key].attrs["Frequency"] != "continuous"
    ):
        errmsg = f"Data {data_key} is not 'continuous'"
        raise ValueError(errmsg)

    # Append the scan index to the path
    scan = data_key + "/" + scan_key

    # Check if the scan index is present
    if scan not in h5f:
        errmsg = f"Scan index {scan_key} not found in {data_key}"
        raise KeyError(errmsg)

    if step_key is None:
        data = {}
        for step_key, dset in h5f[scan].items():
            if dset.size == 0:
                shape = list(dset.shape)
                shape[0] += 1
                dset = np.full(shape, np.nan)

            data[step_key] = np.squeeze(dset)

        return data

    # Append the step index to the path
    step = scan + "/" + step_key

    # Check if the scan index is present
    if step not in h5f:
        errmsg = f"Scan index {step_key} not found in {scan}"
        raise KeyError(errmsg)

    return np.squeeze(h5f[step])
