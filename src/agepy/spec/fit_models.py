from __future__ import annotations

try:
    from numba_stats import (
        norm,
        truncexpon,
        uniform,
        voigt,
    )

except ImportError as e:
    errmsg = "numba_stats required for fitting."
    raise ImportError(errmsg) from e

import numpy as np
import numba as nb
from jacobi import propagate

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray, ArrayLike

__all__ = [
    "SumModel2d",
    "SumModel1d",
    "FitModel2d",
    "Gaussian",
    "Voigt",
    "Constant",
    "Exponential",
]


class SumModel:
    def __init__(self) -> None:
        self.models = {}
        self.par = []

    def update_par(self) -> None:
        self.par = []
        for m in self.models.values():
            for par in m["map"].values():
                if par not in self.par:
                    self.par.append(par)

    def add_model(self, model, idx: int | None = None) -> None:
        if idx is None:
            idx = len(self.models)

        self.models[idx] = {"model": model, "map": {}}
        for par in model.par:
            self.models[idx]["map"][par] = f"{par}_{idx}"

        self.update_par()

    def share_par(self, par: str, idx1: int, idx2: int) -> None:
        if (idx1 not in self.models) or (idx2 not in self.models):
            errmsg = "Unknown model index"
            raise KeyError(errmsg)

        map1 = self.models[idx1]["map"]
        map2 = self.models[idx2]["map"]

        if (par not in map1) or (par not in map2):
            errmsg = f"Unknown parameter {par}"
            raise KeyError(errmsg)

        map1[par] = f"{par}_{idx1}_{idx2}"
        map2[par] = f"{par}_{idx1}_{idx2}"
        self.update_par()

    @property
    def val(self) -> NDArray:
        v = [0] * len(self.par)
        par_dict = dict(zip(self.par, v))

        for mi in self.models.values():
            vi = mi["model"].val

            for vij, par in zip(vi, mi["map"].values()):
                par_dict[par] = vij

        return np.array(list(par_dict.values()))

    @val.setter
    def val(self, v: ArrayLike) -> None:
        par_dict = dict(zip(self.par, v))

        for mi in self.models.values():
            mi["model"].val = np.array(
                [par_dict[par] for par in mi["map"].values()]
            )

    @property
    def err(self) -> NDArray:
        v = [0] * len(self.par)
        par_dict = dict(zip(self.par, v))

        for mi in self.models.values():
            vi = mi["model"].err

            for vij, par in zip(vi, mi["map"].values()):
                par_dict[par] = vij

        return np.array(list(par_dict.values()))

    @err.setter
    def err(self, v: ArrayLike) -> None:
        par_dict = dict(zip(self.par, v))

        for mi in self.models.values():
            mi["model"].val = np.array(
                [par_dict[par] for par in mi["map"].values()]
            )

    @property
    def lim(self) -> dict[str, tuple[float | None, float | None]]:
        v = [(None, None)] * len(self.par)
        par_dict = dict(zip(self.par, v))

        for mi in self.models.values():
            vi = mi["model"].lim

            for orig_par, new_par in mi["map"].items():
                par_dict[new_par] = vi[orig_par]

        return par_dict

    @lim.setter
    def lim(self, v: None) -> None:
        raise NotImplementedError()


class SumModel2d(SumModel):
    def density(self, xe_ye, *par):
        par_dict = dict(zip(self.par, par))
        z = np.zeros_like(xe_ye[0])

        for mi in self.models.values():
            val = [par_dict[p] for p in mi["map"].values()]
            z += mi["model"].density(xe_ye, *val)

        return z

    def integral(self, xe_ye, *par):
        par_dict = dict(zip(self.par, par))
        z = np.zeros_like(xe_ye[0])

        for mi in self.models.values():
            val = [par_dict[p] for p in mi["map"].values()]
            z += mi["model"].integral(xe_ye, *val)

        return z


class SumModel1d(SumModel):
    def density(self, x, *par):
        par_dict = dict(zip(self.par, par))
        y = np.zeros_like(x)

        for mi in self.models.values():
            val = [par_dict[p] for p in mi["map"].values()]
            y += mi["model"].density(x, *val)

        return y

    def integral(self, x, *par):
        par_dict = dict(zip(self.par, par))
        y = np.zeros_like(x)

        for mi in self.models.values():
            val = [par_dict[p] for p in mi["map"].values()]
            y += mi["model"].integral(x, *val)

        return y


class FitModel2d:
    def __init__(self, x: FitModel1d, y: FitModel1d):
        self.x = x
        self.y = y

        if self.x.par[0] == "n" or self.y.par[0] == "n":
            self.par = ["n"]

        else:
            self.par = ["b"]

        if len(self.x.par) > 1:
            for par in self.x.par[1:]:
                self.par.append("x_" + par)

        if len(self.y.par) > 1:
            for par in self.y.par[1:]:
                self.par.append("y_" + par)

        self.xi = ~np.char.startswith(self.par, "y")
        self.yi = ~np.char.startswith(self.par, "x")

    @property
    def val(self) -> NDArray:
        v = np.zeros_like(self.par, dtype=np.float64)

        v[self.xi] = self.x.val
        v[self.yi] = self.y.val

        return v

    @val.setter
    def val(self, v: ArrayLike) -> None:
        v = np.array(v)

        self.x.val = v[self.xi]
        self.y.val = v[self.yi]

    @property
    def err(self) -> NDArray:
        v = np.zeros_like(self.par, dtype=np.float64)

        v[self.xi] = self.x.err
        v[self.yi] = self.y.err

        return v

    @err.setter
    def err(self, v: ArrayLike) -> None:
        v = np.array(v)

        self.x.err = v[self.xi]
        self.y.err = v[self.yi]

    @property
    def lim(self) -> dict[tuple[float | None, float | None]]:
        lim = {}
        xi = 0
        yi = 0

        for i in range(len(self.par)):
            if self.xi[i]:
                lim[self.par[i]] = self.x.lim[self.x.par[xi]]
                xi += 1

            if self.yi[i]:
                lim[self.par[i]] = self.y.lim[self.y.par[yi]]
                yi += 1

        return lim

    @lim.setter
    def lim(self, v: None) -> None:
        raise NotImplementedError()

    def density(self, xe_ye, *par):
        xe, ye = xe_ye
        par = np.array(par)

        x_pdf = self.x.density(xe, *par[self.xi])
        y_pdf = self.y.density(ye, *par[self.yi])

        return x_pdf * y_pdf / par[0]

    def integral(self, xe_ye, *par):
        xe, ye = xe_ye
        par = np.array(par)

        x_cdf = self.x.integral(xe, *par[self.xi])
        y_cdf = self.y.integral(ye, *par[self.yi])

        return x_cdf * y_cdf / par[0]


class FitModel1d:
    name: str = ""
    par: list[str] = []

    def __init__(
        self, val: list[float], **lim: tuple[float | None, float | None]
    ) -> None:
        self.val = np.array(val, dtype=np.float64)
        self.err = np.zeros_like(val, dtype=np.float64)

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

    def set_value(self, par: str, val: float) -> None:
        idx = self.par.index(par)
        self.val[idx] = val
        self.err = np.zeros_like(self.val, dtype=np.float64)

    def value(self, par: str) -> float:
        idx = self.par.index(par)
        return self.val[idx]

    def error(self, par: str) -> float:
        idx = self.par.index(par)
        return self.err[idx]


class Gaussian(FitModel1d):
    name = "Gaussian"
    par = ["n", "loc", "scale"]

    def __init__(self, xr: tuple[float, float]) -> None:
        super().__init__(
            [1, 0, 1],
            n=(0, None),
            loc=(xr[0], xr[1]),
            scale=(0, 0.2 * (xr[1] - xr[0])),
        )

    def density(self, x: ArrayLike, n: float, loc: float, scale: float):
        return n * norm.pdf(x, loc, scale)

    def integral(self, x: ArrayLike, n: float, loc: float, scale: float):
        return n * norm.cdf(x, loc, scale)

    def pdf(self, x: ArrayLike, loc: float, scale: float):
        return norm.pdf(x, loc, scale)

    def cdf(self, x: ArrayLike, loc: float, scale: float):
        return norm.cdf(x, loc, scale)

    def der(self, x: ArrayLike, loc: float, scale: float):
        return norm.pdf(x, loc, scale) * (loc - x) / scale**2


class Voigt(FitModel1d):
    name = "Voigt"
    par = ["n", "gamma", "loc", "scale"]

    def __init__(self, xr: tuple[float, float]) -> None:
        self.xr = xr
        super().__init__(
            [1, 1, 0, 1],
            n=(0, None),
            gamma=(0, None),
            loc=(xr[0], xr[1]),
            scale=(0, 0.2 * (xr[1] - xr[0])),
        )

    def density(
        self, x: ArrayLike, n: float, gamma: float, loc: float, scale: float
    ) -> NDArray:
        return n * voigt.pdf(x, gamma, loc, scale)

    def integral(
        self, x: ArrayLike, n: float, gamma: float, loc: float, scale: float
    ) -> NDArray:
        return n * self.cdf(x, gamma, loc, scale)

    def pdf(
        self, x: ArrayLike, gamma: float, loc: float, scale: float
    ) -> NDArray:
        return voigt.pdf(x, gamma, loc, scale)

    def cdf(
        self, x: ArrayLike, gamma: float, loc: float, scale: float
    ) -> NDArray:
        _x = np.linspace(self.xr[0], self.xr[1], 1000)
        return num_eval_cdf(x, _x, voigt.pdf(_x, gamma, loc, scale))


class Constant(FitModel1d):
    name = "Constant"
    par = ["b"]

    def __init__(self, xr: tuple[float, float]) -> None:
        self.lx = xr[0]
        self.dx = xr[1] - xr[0]
        super().__init__([1], b=(0, None))

    def density(self, x: ArrayLike, b: float):
        return b * uniform.pdf(x, self.lx, self.dx)

    def integral(self, x: ArrayLike, b: float):
        return b * uniform.cdf(x, self.lx, self.dx)

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
    par = ["b", "loc_exp", "scale_exp"]

    def __init__(self, xr: tuple[float, float]) -> None:
        self.xr = xr
        super().__init__(
            [1, 0, 1],
            b=(0, None),
            loc_exp=(-1, 0),
            scale_exp=(None, None),
        )

    def density(
        self, x: ArrayLike, b: float, loc_exp: float, scale_exp: float
    ):
        return b * truncexpon.pdf(
            x, self.xr[0], self.xr[1], loc_exp, scale_exp
        )

    def integral(
        self, x: ArrayLike, b: float, loc_exp: float, scale_exp: float
    ):
        return b * truncexpon.cdf(
            x, self.xr[0], self.xr[1], loc_exp, scale_exp
        )

    def pdf(self, x: ArrayLike, loc_exp: float, scale_exp: float):
        return truncexpon.pdf(x, self.xr[0], self.xr[1], loc_exp, scale_exp)

    def cdf(self, x: ArrayLike, loc_exp: float, scale_exp: float):
        return truncexpon.cdf(x, self.xr[0], self.xr[1], loc_exp, scale_exp)


@nb.njit(parallel=True, fastmath={"reassoc", "contract", "arcp"})
def num_eval_cdf(x, _x, _pdf):
    _y = np.empty_like(_x)

    for i in nb.prange(len(_x)):
        _y[i] = np.trapz(_pdf[: i + 1], x=_x[: i + 1])

    return np.interp(x, _x, _y)
