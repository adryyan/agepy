from __future__ import annotations

import h5py
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray


def load_metro_step(
    h5: h5py.Group,
    data_channel: str,
    scan_idx: str,
    step_idx: str,
) -> NDArray:
    # Check if the data is found
    if data_channel not in h5:
        errmsg = f"Data channel {data_channel} not found"
        raise KeyError(errmsg)

    freq = h5[data_channel].attrs["Frequency"]
    if freq != "continuous":
        errmsg = f"Data channel {data_channel} is not 'continuous'"
        raise ValueError(errmsg)

    scan = data_channel + "/" + scan_idx

    if scan not in h5:
        errmsg = f"Scan index {scan_idx} not found in {data_channel}"
        raise KeyError(errmsg)

    step = scan + "/" + step_idx

    if step not in h5:
        errmsg = f"Step index {step_idx} not found in {scan}"
        raise KeyError(errmsg)

    # Load and return the data
    return np.array(h5[step], dtype=np.float64)


def load_metro_scan(
    h5: h5py.Group,
    data_channel: str,
    scan_idx: str,
) -> tuple[list[str], list[NDArray]]:
    # Check if the data is found
    if data_channel not in h5:
        errmsg = f"Data channel {data_channel} not found"
        raise KeyError(errmsg)

    freq = h5[data_channel].attrs["Frequency"]
    if freq != "continuous":
        errmsg = f"Data channel {data_channel} is not 'continuous'"
        raise ValueError(errmsg)

    scan = data_channel + "/" + scan_idx

    if scan not in h5:
        errmsg = f"Scan index {scan_idx} not found in {data_channel}"
        raise KeyError(errmsg)

    steps = list(h5[scan].keys())
    datasets = [np.array(dset) for dset in h5[scan].values()]

    return steps, datasets
