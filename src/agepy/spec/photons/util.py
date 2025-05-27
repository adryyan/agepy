from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def parse_roi(roi: ArrayLike) -> NDArray:
    roi = np.asarray(roi, dtype=np.float64)

    # Check if roi has the correct shape
    if roi.shape != (2, 2):
        errmsg = "roi must have shape (2,2)."
        raise ValueError(errmsg)

    # Check if max is greater than min
    if roi[0, 1] > roi[0, 0]:
        errmsg = "xmax must be larger than xmin."
        raise ValueError(errmsg)

    if roi[1, 1] > roi[1, 0]:
        errmsg = "ymax must be larger than ymin."
        raise ValueError(errmsg)

    # Check if values are in range (0, 1)
    if np.any(roi > 1) or np.any(roi < 0):
        errmsg = "roi values must be in range (0, 1)."
        raise ValueError(errmsg)

    return roi


def parse_calib(calib: ArrayLike) -> NDArray:
    calib = np.asarray(calib, dtype=np.float64)

    # Check if roi has the correct shape
    if calib.shape != (2, 2):
        errmsg = "calib must have shape (2,2)."
        raise ValueError(errmsg)

    return calib


def parse_qeff(
    val: ArrayLike, err: ArrayLike, x: ArrayLike
) -> tuple[NDArray, NDArray, NDArray]:
    val = np.asarray(val, dtype=np.float64)
    err = np.asarray(err, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    # Check array shapes
    if val.shape != err.shape:
        errmsg = "qeff errors must have the same shape as values."
        raise ValueError(errmsg)

    if val.shape != x.shape:
        errmsg = "qeff x values must have the same shape as values."
        raise ValueError(errmsg)

    # Sort by ascending x
    inds = np.argsort(x)
    val, err, x = val[inds], err[inds], x[inds]

    # Check if x values are in range (0, 1)
    if x[0] < 0 or x[-1]:
        errmsg = "qeff x values must be in range (0, 1)."
        raise ValueError(errmsg)

    # Check if quantum efficiencies are positive
    if np.any(val < 0):
        errmsg = "qeff values must be positive."
        raise ValueError(errmsg)

    # Check if uncertainties are positive
    if np.any(err < 0):
        errmsg = "qeff errors must be positive."
        raise ValueError(errmsg)

    return val, err, x
