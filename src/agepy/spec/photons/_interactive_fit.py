from __future__ import annotations
import warnings

try:
    from numba_stats import (
        bernstein,
        norm,
        truncexpon,
        uniform,
        voigt,
        cruijff,
        crystalball,
        crystalball_ex,
        qgaussian,
        t,
    )

except ImportError as e:
    errmsg = "numba_stats required for fitting."
    raise ImportError(errmsg) from e

import numpy as np
import numba as nb
from scipy.stats import gennorm
from jacobi import propagate

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray


class FitModel:
    name = "Generic"
    par = []

    def __init__(
        self,
        xr: tuple[float, float],
    ) -> None:
        self.xr = xr

        # Fit results
        self.val = None
        self.err = None
        self.cov = None

    def __call__(self, x: NDArray) -> tuple[NDArray, NDArray]:
        if self.cov is None:
            raise ValueError("Fit results are not available.")

        # Propagate the uncertainties
        y, yerr = propagate(lambda par: self.pdf(x, par), self.val, self.cov)

        return y, np.sqrt(np.diag(yerr))

    def value(self, par: str):
        if par not in self.par:
            errmsg = f"Unknown parameter {par}"
            raise ValueError(errmsg)

        idx = self.par.index(par)

        return self.val[idx]

    def error(self, par: str):
        if par not in self.par:
            errmsg = f"Unknown parameter {par}"
            raise ValueError(errmsg)

        idx = self.par.index(par)

        return self.err[idx]

    def start_val(self, n):
        if self.val is not None:
            return {k: v for k, v in zip(self.par, self.val)}

        else:
            return self._start_val(n)


class Gaussian(FitModel):
    name = "Gaussian"
    par = ["s", "loc", "scale"]

    @staticmethod
    def pdf(x, par):
        return par[0] * norm.pdf(x, *par[1:])

    @staticmethod
    def cdf(x, par):
        return par[0] * norm.cdf(x, *par[1:])

    @staticmethod
    def der(x, par):
        return -par[0] * norm.pdf(x, *par[1:]) * (x - par[1]) / par[2] ** 2

    def limits(self, n_max):
        dx = self.xr[1] - self.xr[0]

        return {
            "s": (0, n_max),
            "loc": self.xr,
            "scale": (0.0001 * dx, 0.5 * dx),
        }

    def _start_val(self, n):
        return {
            "s": n,
            "loc": 0.5 * (self.xr[0] + self.xr[1]),
            "scale": 0.05 * (self.xr[1] - self.xr[0]),
        }


class Voigt(FitModel):
    name = "Voigt"
    par = ["s", "gamma", "loc", "scale"]

    @staticmethod
    def pdf(x, par):
        return par[0] * voigt.pdf(x, *par[1:])

    def cdf(self, x, par):
        _x = np.linspace(self.xr[0], self.xr[1], 1000)
        return par[0] * num_eval_cdf(x, _x, voigt.pdf(_x, *par[1:]))

    def limits(self, n_max):
        dx = self.xr[1] - self.xr[0]

        return {
            "s": (0, n_max),
            "gamma": (0.00001 * dx, 0.1 * dx),
            "loc": self.xr,
            "scale": (0.0001 * dx, 0.5 * dx),
        }

    def _start_val(self, n):
        dx = self.xr[1] - self.xr[0]

        return {
            "s": n,
            "gamma": 0.01 * dx,
            "loc": 0.5 * (self.xr[0] + self.xr[1]),
            "scale": 0.05 * (self.xr[1] - self.xr[0]),
        }


class DoubleGaussian(FitModel):
    name = "Gaussian + Gaussian"
    par = ["s", "loc", "scale_a", "scale_b", "ratio"]

    @staticmethod
    def pdf(x, par):
        return par[0] * (
            par[4] * norm.pdf(x, par[1], par[2])
            + (1 - par[4]) * norm.pdf(x, par[1], par[3])
        )

    def cdf(self, x, par):
        return par[0] * (
            par[4] * norm.cdf(x, par[1], par[2])
            + (1 - par[4]) * norm.cdf(x, par[1], par[3])
        )

    def limits(self, n_max):
        dx = self.xr[1] - self.xr[0]

        return {
            "s": (0, n_max),
            "loc": self.xr,
            "scale_a": (0.001 * dx, dx),
            "scale_b": (0.0001 * dx, 0.5 * dx),
            "ratio": (0, 1),
        }

    def _start_val(self, n):
        return {
            "s": n,
            "loc": 0.5 * (self.xr[0] + self.xr[1]),
            "scale_a": 0.05 * (self.xr[1] - self.xr[0]),
            "scale_b": 0.5 * (self.xr[1] - self.xr[0]),
            "ratio": 0.9,
        }


class VoigtBox(FitModel):
    name = "Voigt + Box"
    par = ["s", "gamma", "loc", "scale", "ratio", "width"]

    @staticmethod
    def pdf(x, par):
        xmin = par[2] - 0.5 * par[5]

        return par[0] * (
            par[4] * uniform.pdf(x, xmin, par[5])
            + (1 - par[4]) * voigt.pdf(x, par[1], par[2], par[3])
        )

    def cdf(self, x, par):
        xmin = par[2] - 0.5 * par[5]

        _x = np.linspace(self.xr[0], self.xr[1], 1000)

        return par[0] * (
            par[4] * uniform.cdf(x, xmin, par[5])
            + (1 - par[4])
            * num_eval_cdf(x, _x, voigt.pdf(_x, par[1], par[2], par[3]))
        )

    def limits(self, n_max):
        dx = self.xr[1] - self.xr[0]

        return {
            "s": (0, n_max),
            "gamma": (0.00001 * dx, 0.1 * dx),
            "loc": self.xr,
            "scale": (0.0001 * dx, dx),
            "ratio": (0, 1),
            "width": (0.0001 * dx, 0.5 * dx),
        }

    def _start_val(self, n):
        dx = self.xr[1] - self.xr[0]

        return {
            "s": n,
            "gamma": 0.01 * dx,
            "loc": 0.5 * (self.xr[0] + self.xr[1]),
            "scale": 0.05 * (self.xr[1] - self.xr[0]),
            "ratio": 0.2,
            "width": 0.05 * (self.xr[1] - self.xr[0]),
        }


class VoigtGaussian(FitModel):
    name = "Voigt + Gaussian"
    par = ["s", "gamma", "loc", "scale_voigt", "scale_norm", "ratio"]

    @staticmethod
    def pdf(x, par):
        return par[0] * (
            par[5] * voigt.pdf(x, *par[1:4])
            + (1 - par[5]) * norm.pdf(x, par[2], par[4])
        )

    def cdf(self, x, par):
        _x = np.linspace(self.xr[0], self.xr[1], 1000)
        return par[0] * (
            par[5] * num_eval_cdf(x, _x, voigt.pdf(_x, *par[1:4]))
            + (1 - par[5]) * norm.cdf(x, par[2], par[4])
        )

    def limits(self, n_max):
        dx = self.xr[1] - self.xr[0]

        return {
            "s": (0, n_max),
            "gamma": (0.00001 * dx, 0.1 * dx),
            "loc": self.xr,
            "scale_voigt": (0.0001 * dx, 0.5 * dx),
            "scale_norm": (0.0001 * dx, 0.5 * dx),
            "ratio": (0, 1),
        }

    def _start_val(self, n):
        dx = self.xr[1] - self.xr[0]

        return {
            "s": n,
            "gamma": 0.01 * dx,
            "loc": 0.5 * (self.xr[0] + self.xr[1]),
            "scale_voigt": 0.05 * (self.xr[1] - self.xr[0]),
            "scale_norm": 0.05 * (self.xr[1] - self.xr[0]),
            "ratio": 0.5,
        }


class QGaussian(FitModel):
    name = "Q-Gaussian"
    par = ["s", "q", "loc", "scale"]

    @staticmethod
    def pdf(x, par):
        if par[1] < 1:
            par[1] = 1
            wrnmsg = "q cannot be smaller than 1. Setting q=1."
            warnings.warn(wrnmsg, stacklevel=1)

        if par[1] > 3:
            par[1] = 3
            wrnmsg = "q cannot be larger than 3. Setting q=3."
            warnings.warn(wrnmsg, stacklevel=1)

        return par[0] * qgaussian.pdf(x, *par[1:])

    def cdf(self, x, par):
        if par[1] < 1:
            par[1] = 1
            wrnmsg = "q cannot be smaller than 1. Setting q=1."
            warnings.warn(wrnmsg, stacklevel=1)

        if par[1] > 3:
            par[1] = 3
            wrnmsg = "q cannot be larger than 3. Setting q=3."
            warnings.warn(wrnmsg, stacklevel=1)

        return par[0] * qgaussian.cdf(x, *par[1:])

    def limits(self, n_max):
        dx = self.xr[1] - self.xr[0]

        return {
            "s": (0, n_max),
            "q": (1, 3),
            "loc": self.xr,
            "scale": (0.0001 * dx, 0.5 * dx),
        }

    def _start_val(self, n):
        return {
            "s": n,
            "q": 2,
            "loc": 0.5 * (self.xr[0] + self.xr[1]),
            "scale": 0.05 * (self.xr[1] - self.xr[0]),
        }


class GeneralizedGaussian(FitModel):
    name = "Generalized Gaussian"
    par = ["s", "beta", "loc", "scale"]

    @staticmethod
    def pdf(x, par):
        return par[0] * gennorm.pdf(x, *par[1:])

    def cdf(self, x, par):
        return par[0] * gennorm.cdf(x, *par[1:])

    def limits(self, n_max):
        dx = self.xr[1] - self.xr[0]

        return {
            "s": (0, n_max),
            "beta": (0, 50),
            "loc": self.xr,
            "scale": (0.0001 * dx, 0.5 * dx),
        }

    def _start_val(self, n):
        return {
            "s": n,
            "beta": 2,
            "loc": 0.5 * (self.xr[0] + self.xr[1]),
            "scale": 0.05 * (self.xr[1] - self.xr[0]),
        }


class Studentst(FitModel):
    name = "Student's t"
    par = ["s", "df", "loc", "scale"]

    @staticmethod
    def pdf(x, par):
        return par[0] * t.pdf(x, *par[1:])

    def cdf(self, x, par):
        return par[0] * t.cdf(x, *par[1:])

    def limits(self, n_max):
        dx = self.xr[1] - self.xr[0]

        return {
            "s": (0, n_max),
            "df": (0, 50),
            "loc": self.xr,
            "scale": (0.0001 * dx, 0.5 * dx),
        }

    def _start_val(self, n):
        return {
            "s": n,
            "df": 1,
            "loc": 0.5 * (self.xr[0] + self.xr[1]),
            "scale": 0.05 * (self.xr[1] - self.xr[0]),
        }


class Cruijff(FitModel):
    name = "Cruijff"
    par = ["s", "beta_left", "beta_right", "loc", "scale_left", "scale_right"]

    @staticmethod
    def pdf(x, par):
        return par[0] * cruijff.density(x, *par[1:])

    def cdf(self, x, par):
        _x = np.linspace(self.xr[0], self.xr[1], 1000)
        return par[0] * num_eval_cdf(x, _x, cruijff.density(_x, *par[1:]))

    def limits(self, n_max):
        dx = self.xr[1] - self.xr[0]

        return {
            "s": (0, n_max),
            "beta_left": (0, 1),
            "beta_right": (0, 1),
            "loc": self.xr,
            "scale_left": (0.0001 * dx, 0.5 * dx),
            "scale_right": (0.0001 * dx, 0.5 * dx),
        }

    def _start_val(self, n):
        dx = self.xr[1] - self.xr[0]

        return {
            "s": n,
            "beta_left": 0.1,
            "beta_right": 0.1,
            "loc": 0.5 * (self.xr[0] + self.xr[1]),
            "scale_left": 0.05 * dx,
            "scale_right": 0.05 * dx,
        }


class CrystalBall(FitModel):
    name = "CrystalBall"
    par = ["s", "beta", "m", "loc", "scale"]

    @staticmethod
    def pdf(x, par):
        return par[0] * crystalball.pdf(x, *par[1:])

    @staticmethod
    def cdf(x, par):
        return par[0] * crystalball.cdf(x, *par[1:])

    def limits(self, n_max):
        dx = self.xr[1] - self.xr[0]

        return {
            "s": (0, n_max),
            "beta": (0, 5),
            "m": (1, 10),
            "loc": self.xr,
            "scale": (0.0001 * dx, 0.5 * dx),
        }

    def _start_val(self, n):
        dx = self.xr[1] - self.xr[0]

        return {
            "s": n,
            "beta": 1,
            "m": 2,
            "loc": 0.5 * (self.xr[0] + self.xr[1]),
            "scale": 0.05 * dx,
        }


class CrystalBallEx(FitModel):
    name = "CrystalBallEx"
    par = [
        "s",
        "beta_left",
        "m_left",
        "scale_left",
        "beta_right",
        "m_right",
        "scale_right",
        "loc",
    ]

    @staticmethod
    def pdf(x, par):
        return par[0] * crystalball_ex.pdf(x, *par[1:])

    @staticmethod
    def cdf(x, par):
        return par[0] * crystalball_ex.cdf(x, *par[1:])

    def limits(self, n_max):
        dx = self.xr[1] - self.xr[0]

        return {
            "s": (0, n_max),
            "beta_left": (0, 5),
            "m_left": (1, 10),
            "scale_left": (0.0001 * dx, 0.5 * dx),
            "beta_right": (0, 5),
            "m_right": (1, 10),
            "scale_right": (0.0001 * dx, 0.5 * dx),
            "loc": self.xr,
        }

    def _start_val(self, n):
        dx = self.xr[1] - self.xr[0]

        return {
            "s": n,
            "beta_left": 1,
            "m_left": 2,
            "scale_left": 0.05 * dx,
            "beta_right": 1,
            "m_right": 2,
            "scale_right": 0.05 * dx,
            "loc": 0.5 * (self.xr[0] + self.xr[1]),
        }


class Bernstein(FitModel):
    par = ["b_ij"]

    def __init__(
        self,
        deg: int,
        xr: tuple[float, float],
    ) -> None:
        super().__init__(xr)

        self.deg = deg
        self.par = [f"b_{i}{deg}" for i in range(deg + 1)]

    def pdf(self, x, par):
        return bernstein.density(x, par, *self.xr)

    def cdf(self, x, par):
        return bernstein.integral(x, par, *self.xr)

    def limits(self):
        return {f"b_{i}{self.deg}": (0, None) for i in range(self.deg + 1)}

    def start_val(self):
        return {f"b_{i}{self.deg}": 1 for i in range(self.deg + 1)}


class Constant(FitModel):
    par = ["b"]

    def pdf(self, x, par):
        return par[0] * uniform.pdf(x, self.xr[0], self.xr[1] - self.xr[0])

    def cdf(self, x, par):
        return par[0] * uniform.cdf(x, self.xr[0], self.xr[1] - self.xr[0])

    def der(self, x, par):
        return np.zeros_like(x)

    def limits(self):
        return {"b": (0, None)}

    def start_val(self):
        return {"b": 1}


class Exponential(FitModel):
    par = ["b", "loc_expon", "scale_expon"]

    def pdf(self, x, par):
        return par[0] * truncexpon.pdf(x, self.xr[0], self.xr[1], *par[1:])

    def cdf(self, x, par):
        return par[0] * truncexpon.cdf(x, self.xr[0], self.xr[1], *par[1:])

    def limits(self):
        return {
            "b": (0, None),
            "loc_expon": (-1, 0),
            "scale_expon": (-100, 100),
        }

    def start_val(self):
        return {"b": 1, "loc_expon": -0.5, "scale_expon": 1}


@nb.njit(parallel=True, fastmath={"reassoc", "contract", "arcp"})
def num_eval_cdf(x, _x, _pdf):
    _y = np.empty_like(_x)

    for i in nb.prange(len(_x)):
        _y[i] = np.trapz(_pdf[: i + 1], x=_x[: i + 1])

    return np.interp(x, _x, _y)
