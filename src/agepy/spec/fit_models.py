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
import xarray as xr
import numba as nb
from scipy.stats import gennorm
from jacobi import propagate

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray, ArrayLike


class FitParam:
    def __init__(
        self,
        name: str,
        val: float = 0,
        lim: tuple[float | None, float | None] = (None, None),
    ) -> None:
        self.name = str(name)
        self.val = float(val)
        self.err = 0.0
        self.lim = lim


class YieldParam(FitParam):
    def __init__(self) -> None:
        super().__init__("n", val=1, lim=(0, None))


class LocParam(FitParam):
    def __init__(self) -> None:
        super().__init__("loc", val=0, lim=(None, None))


class ScaleParam(FitParam):
    def __init__(self) -> None:
        super().__init__("scale", val=1, lim=(0, None))


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
        z = np.zeros_like(xe_ye[0])

        for m in self.models:
            val = [par_dict[p] for p in self.par_map[m].values()]
            z += self.models[m].density(xe_ye, *val)

        return z

    def integral(self, xe_ye, *par):
        par_dict = dict(zip(self.par, par))
        z = np.zeros_like(xe_ye[0])

        for m in self.models:
            val = [par_dict[p] for p in self.par_map[m].values()]
            z += self.models[m].integral(xe_ye, *val)

        return z


class SumModel1d(SumModel):
    def density(self, x, *par):
        par_dict = dict(zip(self.par, par))
        y = np.zeros_like(x)

        for m in self.models:
            val = [par_dict[p] for p in self.par_map[m].values()]
            y += self.models[m].density(x, *val)

        return z

    def integral(self, x, *par):
        par_dict = dict(zip(self.par, par))
        y = np.zeros_like(x)

        for m in self.models:
            val = [par_dict[p] for p in self.par_map[m].values()]
            y += self.models[m].integral(x, *val)

        return z


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

    def __init__(
        self, val: list[float], **lim: tuple[float | None, float | None]
    ) -> None:
        self.val = xr.DataArray(val, coords=[("par", self.par)])
        self.err = xr.DataArray(np.zeros_like(val), coords=[("par", self.par)])

        self.lim = {}
        for par in self.par:
            if par in lim:
                self.lim[par] = lim[par]

            else:
                self.lim[par] = (None, None)

    def __call__(self, x: NDArray) -> tuple[NDArray, NDArray]:
        y, err = propagate(
            lambda par: self.density(x, par), self.val, self.err
        )

        return y, np.sqrt(np.diag(err))

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
    par = ["n", "loc", "scale"]

    def __init__(self) -> None:
        super().__init__([1, 0, 1], n=(0, None), scale=(0, None))

    def density(self, x: ArrayLike):
        return self.val[0] * norm.pdf(x, self.val[1:])

    def integral(self, x: ArrayLike, s: float, loc: float, scale: float):
        return self.val[0] * norm.cdf(x, self.val[1:])

    def pdf(self, x: ArrayLike):
        return norm.pdf(x, self.val[1:])

    def cdf(self, x: ArrayLike):
        return norm.cdf(x, self.val[1:])

    def der(self, x: ArrayLike):
        return norm.pdf(x, self.val[1:]) * (self.val[1] - x) / self.val[2] ** 2


class Voigt(FitModel1d):
    name = "Voigt"
    par = ["s", "gamma", "loc", "scale"]

    def __init__(self) -> None:
        super().__init__(
            [1, 1, 0, 1], n=(0, None), gamma=(0, None), scale=(0, None)
        )

    def density(self, x: ArrayLike) -> NDArray:
        return self.val[0] * voigt.pdf(x, self.val[1:])

    def integral(self, x: ArrayLike) -> NDArray:
        return self.val[0] * self.cdf(x, self.val[1:])

    def pdf(self, x: ArrayLike) -> NDArray:
        return voigt.pdf(x, self.val[1:])

    def cdf(self, x: ArrayLike) -> NDArray:
        _x = np.linspace(x[0], x[-1], 1000)
        return num_eval_cdf(x, _x, voigt.pdf(_x, self.val[1:]))


class Constant(FitModel1d):
    name = "Constant"
    par = ["b"]

    def __init__(self, xr: tuple[float, float]) -> None:
        self.lx = xr[0]
        self.dx = xr[1] - xr[0]
        super().__init__([1], b=(0, None))

    def density(self, x: ArrayLike):
        return self.val[0] * uniform.pdf(x, self.lx, self.dx)

    def integral(self, x: ArrayLike):
        return self.val[0] * uniform.cdf(x, self.lx, self.dx)

    def grad(self, x: ArrayLike):
        return np.zeros_like(x)

    def pdf(self, x: ArrayLike):
        return uniform.pdf(x, self.lx, self.dx)

    def cdf(self, x: ArrayLike):
        return uniform.cdf(x, self.lx, self.dx)

    def der(self, x: ArrayLike):
        return np.zeros_like(x)


class Exponential(FitModel1d):
    name = "Exponential"
    par = ["b", "loc_expon", "scale_expon"]

    def __init__(self, xr: tuple[float, float]) -> None:
        self.xr = xr
        super().__init__(
            [1, -1, 1],
            b=(0, None),
            loc_expon=(-1, 0),
            scale_expon=(None, None),
        )

    def density(self, x: ArrayLike):
        return self.val[0] * truncexpon.pdf(
            x, self.xr[0], self.xr[1], self.val[1:]
        )

    def integral(self, x: ArrayLike):
        return self.val[0] * truncexpon.cdf(
            x, self.xr[0], self.xr[1], self.val[1:]
        )

    def pdf(self, x: ArrayLike):
        return truncexpon.pdf(x, self.xr[0], self.xr[1], self.val[1:])

    def cdf(self, x: ArrayLike):
        return truncexpon.cdf(x, self.xr[0], self.xr[1], self.val[1:])


@nb.njit(parallel=True, fastmath={"reassoc", "contract", "arcp"})
def num_eval_cdf(x, _x, _pdf):
    _y = np.empty_like(_x)

    for i in nb.prange(len(_x)):
        _y[i] = np.trapz(_pdf[: i + 1], x=_x[: i + 1])

    return np.interp(x, _x, _y)
