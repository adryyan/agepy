from __future__ import annotations

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

import warnings
import numpy as np
import numba as nb
from scipy.stats import gennorm
from jacobi import propagate

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray, ArrayLike


class SumModel:
    def __init__(self) -> None:
        self.models = {}
        self.par_map = {}
        self.par = []

    def update_par(self) -> None:
        self.par = []

        for par_map in self.par_map.values():
            for par in par_map.values():
                if par not in self.par:
                    self.par.append(par)

    def add_model(self, model) -> None:
        model_idx = len(self.models)

        self.models[model_idx] = model
        self.par_map[model_idx] = {}

        for par in model.par:
            par_new = f"{par}_{model_idx}"
            self.par_map[model_idx][par] = par_new

        self.update_par()

    def merge_par(self, par: str, model1: int, model2: int) -> None:
        par_new = f"{par}_{model1}_{model2}"
        self.par_map[model1][par] = par_new
        self.par_map[model2][par] = par_new

        self.update_par()

    @property
    def val(self) -> NDArray:
        v = [0] * len(self.par)
        par_dict = dict(zip(self.par, v))

        for mi in self.models:
            vi = self.models[mi].val

            for vij, par in zip(vi, self.par_map[mi].values()):
                par_dict[par] = vij

        return np.array(par_dict.values())

    @val.setter
    def val(self, v: ArrayLike) -> None:
        par_dict = dict(zip(self.par, v))

        for mi in self.models:
            self.models[mi].val = np.array(
                [par_dict[par] for par in self.par_map[mi].values()]
            )

    @property
    def err(self) -> NDArray:
        v = [0] * len(self.par)
        par_dict = dict(zip(self.par, v))

        for mi in self.models:
            vi = self.models[mi].err

            for vij, par in zip(vi, self.par_map[mi].values()):
                par_dict[par] = vij

        return np.array(par_dict.values())

    @err.setter
    def err(self, v: ArrayLike) -> None:
        par_dict = dict(zip(self.par, v))

        for mi in self.models:
            self.models[mi].err = np.array(
                [par_dict[par] for par in self.par_map[mi].values()]
            )

    @property
    def lim(self) -> dict[str, tuple[float | None, float | None]]:
        v = [(None, None)] * len(self.par)
        par_dict = dict(zip(self.par, v))

        for mi in self.models:
            vi = self.models[mi].lim

            for orig_par, new_par in self.par_map[mi].items():
                par_dict[new_par] = vi[orig_par]

    @lim.setter
    def lim(self, v: None) -> None:
        raise NotImplementedError()


class SumModel2d(SumModel):
    def density(self, xe_ye, *par):
        par_dict = dict(zip(self.par, par))
        z = self.np.zeros_like(xe_ye[0])

        for m in self.models:
            val = [par_dict[p] for p in self.par_map[m].values()]
            z += self.models[m].density(xe_ye, *val)

    def integral(self, xe_ye, *par):
        par_dict = dict(zip(self.par, par))
        z = self.np.zeros_like(xe_ye[0])

        for m in self.models:
            val = [par_dict[p] for p in self.par_map[m].values()]
            z += self.models[m].integral(xe_ye, *val)


class SumModel1d(SumModel):
    def density(self, x, *par):
        par_dict = dict(zip(self.par, par))
        y = self.np.zeros_like(x)

        for m in self.models:
            val = [par_dict[p] for p in self.par_map[m].values()]
            y += self.models[m].density(x, *val)

    def integral(self, x, *par):
        par_dict = dict(zip(self.par, par))
        y = self.np.zeros_like(x)

        for m in self.models:
            val = [par_dict[p] for p in self.par_map[m].values()]
            y += self.models[m].integral(x, *val)


class FitModel2d:
    def __init__(self, x_model: FitModel1d, y_model: FitModel1d):
        self.x = x_model
        self.y = y_model

        self.par = []

        self.yld = self.x.yld or self.y.yld
        if self.yld:
            self.par = ["s"]

        self.xi = int(self.yld)
        self.yi = self.xi + len(self.x.par) - int(self.x.yld)

        for par in self.x.par[int(self.x.yld) :]:
            self.par.append("x_" + par)

        for par in self.y.par[int(self.y.yld) :]:
            self.par.append("y_" + par)

    @property
    def val(self) -> NDArray:
        v = np.zeros_like(self.par)

        v[self.xi : self.yi] = self.x.val[int(self.x.yld) :]
        v[self.yi :] = self.y.val[int(self.y.yld) :]

        if self.x.yld:
            v[0] = self.x.val[0]

        if self.y.yld:
            v[0] = self.y.val[0]

        return v

    @val.setter
    def val(self, v: ArrayLike) -> None:
        v = np.array(v)

        self.x.val[int(self.x.yld) :] = v[self.xi : self.yi]
        self.y.val[int(self.y.yld) :] = v[self.yi :]

        if self.x.yld:
            self.x.val[0] = v[0]

        if self.y.yld:
            self.y.val[0] = v[0]

    @property
    def err(self) -> NDArray:
        v = np.zeros_like(self.par)

        v[self.xi : self.yi] = self.x.err[int(self.x.yld) :]
        v[self.yi :] = self.y.err[int(self.y.yld) :]

        if self.x.yld:
            v[0] = self.x.err[0]

        if self.y.yld:
            v[0] = self.y.err[0]

        return v

    @err.setter
    def err(self, v: ArrayLike) -> None:
        v = np.array(v)

        self.x.err[int(self.x.yld) :] = v[self.xi : self.yi]
        self.y.err[int(self.y.yld) :] = v[self.yi :]

        if self.x.yld:
            self.x.err[0] = v[0]

        if self.y.yld:
            self.y.err[0] = v[0]

    @property
    def lim(self) -> dict[tuple[float | None, float | None]]:
        lim = {}

        if self.x.yld:
            lim["s"] = self.x.lim["s"]

        if self.y.yld:
            lim["s"] = self.y.lim["s"]

        for i in range(self.xi, self.yi):
            par = self.x.par[i - self.xi + int(self.x.yld)]
            lim[self.par[i]] = self.x.lim[par]

        for i in range(self.yi, len(self.par)):
            par = self.y.par[i - self.yi + int(self.y.yld)]
            lim[self.par[i]] = self.y.lim[par]

    @lim.setter
    def lim(self, v: None) -> None:
        raise NotImplementedError()

    def density(self, xe_ye, *par):
        xe, ye = xe_ye

        if self.yld:
            yld = par[0]

        else:
            yld = 1

        return (
            yld
            * self.x.pdf(xe, *par[self.xi : self.yi])
            * self.y.pdf(ye, *par[self.yi :])
        )

    def integral(self, xe_ye, *par):
        xe, ye = xe_ye

        if self.yld:
            yld = par[0]

        else:
            yld = 1

        return (
            yld
            * self.x.cdf(xe, *par[self.xi : self.yi])
            * self.y.cdf(ye, *par[self.yi :])
        )


class FitModel1d:
    name: str
    par: list[str]
    yld: bool

    _val: list[float]
    _lim: dict[tuple[float | None, float | None]]

    def __init__(self) -> None:
        self.val = np.array(self._val)
        self.err = np.zeros_like(self.val)
        self.cov = np.zeros_like(self.val)

        self.lim = self._lim.copy()

    def __call__(self, x: NDArray) -> tuple[NDArray, NDArray]:
        y, err = propagate(
            lambda par: self.density(x, par), self.val, self.err
        )

        return y, np.sqrt(np.diag(err))

    def params(self, yld: bool = True):
        if not yld and self.yld:
            return self.par[1:]

        else:
            return self.par[:]

    def value(self, par: str):
        if par not in self.par:
            errmsg = f"Unknown parameter {par}"
            raise ValueError(errmsg)

        idx = self.par.index(par)

        return self.val[idx]

    def set_value(self, par: str, val: float) -> None:
        idx = self.par.index(par)
        self.val[idx] = val

    def error(self, par: str):
        if par not in self.par:
            errmsg = f"Unknown parameter {par}"
            raise ValueError(errmsg)

        idx = self.par.index(par)

        return self.err[idx]


class Gaussian(FitModel1d):
    name = "Gaussian"
    par = ["s", "loc", "scale"]
    yld = True

    _val = [1, 0, 1]
    _lim = {
        "s": (0, None),
        "loc": (None, None),
        "scale": (0, None),
    }

    def density(self, x: ArrayLike, s: float, loc: float, scale: float):
        return s * norm.pdf(x, loc, scale)

    def integral(self, x: ArrayLike, s: float, loc: float, scale: float):
        return s * norm.cdf(x, loc, scale)

    def pdf(self, x: ArrayLike, loc: float, scale: float):
        return norm.pdf(x, loc, scale)

    def cdf(self, x: ArrayLike, loc: float, scale: float):
        return norm.cdf(x, loc, scale)

    def der(self, x: ArrayLike, loc: float, scale: float):
        return norm.pdf(x, loc, scale) * (loc - x) / scale**2


class Voigt(FitModel1d):
    name = "Voigt"
    par = ["s", "gamma", "loc", "scale"]
    yld = True

    _val = [1, 1, 0, 1]
    _lim = {
        "s": (0, None),
        "gamma": (0, None),
        "loc": (None, None),
        "scale": (0, None),
    }

    def density(
        self, x: ArrayLike, s: float, gamma: float, loc: float, scale: float
    ) -> NDArray:
        return s * voigt.pdf(x, gamma, loc, scale)

    def integral(
        self, x: ArrayLike, s: float, gamma: float, loc: float, scale: float
    ) -> NDArray:
        return s * self.cdf(x, gamma, loc, scale)

    def pdf(
        self, x: ArrayLike, gamma: float, loc: float, scale: float
    ) -> NDArray:
        return voigt.pdf(x, gamma, loc, scale)

    def cdf(
        self, x: ArrayLike, gamma: float, loc: float, scale: float
    ) -> NDArray:
        _x = np.linspace(self.xr[0], self.xr[1], 1000)
        return num_eval_cdf(x, _x, voigt.pdf(_x, gamma, loc, scale))


class DoubleGaussian(FitModel1d):
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


class VoigtBox(FitModel1d):
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


class VoigtGaussian(FitModel1d):
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


class QGaussian(FitModel1d):
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


class GeneralizedGaussian(FitModel1d):
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


class Studentst(FitModel1d):
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


class Cruijff(FitModel1d):
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


class CrystalBall(FitModel1d):
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


class CrystalBallEx(FitModel1d):
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


class Bernstein(FitModel1d):
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


class Constant(FitModel1d):
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


class Exponential(FitModel1d):
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
