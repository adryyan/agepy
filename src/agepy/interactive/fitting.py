from __future__ import annotations
from typing import TYPE_CHECKING
import warnings
from functools import partial
# Import PySide6 / PyQt6 modules
from . util import import_qt_binding
qt_binding, QtWidgets, QtCore, QtGui = import_qt_binding()
# Import the modules for the fitting
from . util import import_iminuit, import_iminuit_interactive
Minuit, cost = import_iminuit()
make_widget = import_iminuit_interactive()
import numpy as np
from numba_stats import (
    bernstein,
    truncnorm,
    norm,
    truncexpon,
    uniform,
    voigt,
    cruijff,
    crystalball,
    crystalball_ex,
    qgaussian
)
import numba as nb
import jax
from jax import numpy as jnp
from jax.scipy.stats import norm as jnorm
from jax.scipy.stats import expon as jexpon
from resample.bootstrap import variance as bvar
from jacobi import propagate
# Import modules for type hints
import matplotlib.pyplot as plt
if TYPE_CHECKING:
    from typing import Sequence, Tuple, Dict, Union
    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D
    from numpy.typing import ArrayLike, NDArray

__all__ = ["PhotonSpectrumFit"]


class Fit2Signals1Background(QtWidgets.QWidget):
    def __init__(self,
        xr: Tuple[float, float],
        sig: Union[str, Tuple[str, str]] = "Gaussian",
        bkg: str = "None",
        parent=None
    ) -> None:
        # Initialize the models
        self._init_model_list()
        if isinstance(sig, str):
            if sig not in self.sig_models.keys():
                raise ValueError(f"Signal model {sig} not found.")
            sig = [sig, "None"]
        else:
            if sig[0] not in self.sig_models.keys():
                raise ValueError(f"Signal model {sig[0]} not found.")
            if sig[1] not in list(self.sig_models.keys()) + ["None"]:
                raise ValueError(f"Signal model {sig[1]} not found.")
        if bkg not in self.bkg_models.keys():
            raise ValueError(f"Background model {bkg} not found.")
        self.params = {}
        self.limits = {}
        self._numint_cdf = None
        self.xr = xr

        # Initialize the parent class
        super().__init__(parent)
        self.setWindowTitle("Spectrum Fit")
        self.resize(1350, 850)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.setFont(font)

        self.layout = QtWidgets.QGridLayout(self)

        # Create dummy plot
        self.fit_widget = QtWidgets.QWidget()
        self.layout.addWidget(self.fit_widget, 0, 0, 1, 3)

        # Create signal model selection widget
        self.signal_group = QtWidgets.QGroupBox("Signal Model")
        self.signal_layout = QtWidgets.QHBoxLayout(self.signal_group)
        self.sig_comp1 = QtWidgets.QComboBox()
        self.sig_comp1.addItems(self.sig_models.keys())
        self.sig_comp1.setCurrentIndex(list(self.sig_models.keys()).index(sig[0]))
        self.sig_comp1.currentIndexChanged.connect(self.prepare_fit)
        self.signal_layout.addWidget(self.sig_comp1)
        self.sig_comp2 = QtWidgets.QComboBox()
        self.sig_comp2.addItems(list(self.sig_models.keys()) + ["None"])
        self.sig_comp2.setCurrentIndex((list(self.sig_models.keys()) + ["None"]).index(sig[1]))
        self.sig_comp2.currentIndexChanged.connect(self.prepare_fit)
        self.signal_layout.addWidget(self.sig_comp2)
        self.layout.addWidget(self.signal_group, 1, 0, 1, 2)

        # Create background model selection widget
        self.background_group = QtWidgets.QGroupBox("Background Model")
        self.background_layout = QtWidgets.QHBoxLayout(self.background_group)
        self.bkg_comp = QtWidgets.QComboBox()
        self.bkg_comp.addItems(self.bkg_models.keys())
        self.bkg_comp.setCurrentIndex(list(self.bkg_models.keys()).index(bkg))
        self.bkg_comp.currentIndexChanged.connect(self.prepare_fit)
        self.background_layout.addWidget(self.bkg_comp)
        self.layout.addWidget(self.background_group, 1, 2, 1, 1)

        # Create the initial fit widget
        self.prepare_fit()

    def prepare_fit(self) -> None:
        sig1 = self.sig_comp1.currentText()
        sig2 = self.sig_comp2.currentText()
        bkg = self.bkg_comp.currentText()
        # Remember current parameters and limits
        _params_prev = self.params.copy()
        # Initialize the signal and background models
        if bkg == "None" and sig2 == "None":
            _model, _integral, _jax_model, self.params, self.limits = self.sig_models[sig1]()
            self.params = {f"{k}1": v for k, v in self.params.items()}
            self.limits = {f"{k}1": v for k, v in self.limits.items()}

            def model(x, *args):
                return _model(x, args)

            self._model = model

            def integral(x, *args):
                return _integral(x, args)

            self._integral = integral

            if _jax_model is None:
                self._jax_model = None
            else:
                def jax_model(x, *args):
                    return _jax_model(x, args)

                self._jax_model = jax_model
        elif sig2 == "None" or bkg == "None":
            _model1, _integral1, _jax_model1, _params1, _limits1 = self.sig_models[sig1]()
            _params1 = {f"{k}1": v for k, v in _params1.items()}
            _limits1 = {f"{k}1": v for k, v in _limits1.items()}
            if bkg == "None":
                if "loc1" in _params1:
                    _params1["loc1"] = self.xr[0] + (self.xr[1] - self.xr[0]) * 0.35
                _model2, _integral2, _jax_model2, _params2, _limits2 = self.sig_models[sig2]()
                if "loc" in _params2:
                    _params2["loc"] = self.xr[0] + (self.xr[1] - self.xr[0]) * 0.65
                _params2 = {f"{k}2": v for k, v in _params2.items()}
                _limits2 = {f"{k}2": v for k, v in _limits2.items()}
            else:
                _model2, _integral2, _jax_model2, _params2, _limits2 = self.bkg_models[bkg]()
            # Combine the parameters and limits
            self.params = dict(_params1, **_params2)
            self.limits = dict(_limits1, **_limits2)
            # Define the combined model function
            idx = len(_params1)

            def model(x, *args):
                return _model1(x, args[:idx]) + _model2(x, args[idx:])

            self._model = model
            # Define the combined integral function

            def integral(x, *args):
                return _integral1(x, args[:idx]) + _integral2(x, args[idx:])

            self._integral = integral
            # Define the combined derivative function
            if _jax_model1 is None or _jax_model2 is None:
                self._jax_model = None
            else:

                def jax_model(x, *args):
                    return _jax_model1(x, args[:idx]) + _jax_model2(x, args[idx:])

                self._jax_model = jax_model
        else:
            _model1, _integral1, _jax_model1, _params1, _limits1 = self.sig_models[sig1]()
            if "loc" in _params1:
                _params1["loc"] = self.xr[0] + (self.xr[1] - self.xr[0]) * 0.35
            _params1 = {f"{k}1": v for k, v in _params1.items()}
            _limits1 = {f"{k}1": v for k, v in _limits1.items()}
            _model2, _integral2, _jax_model2, _params2, _limits2 = self.sig_models[sig2]()
            if "loc" in _params2:
                _params2["loc"] = self.xr[0] + (self.xr[1] - self.xr[0]) * 0.65
            _params2 = {f"{k}2": v for k, v in _params2.items()}
            _limits2 = {f"{k}2": v for k, v in _limits2.items()}
            _model3, _integral3, _jax_model3, _params3, _limits3 = self.bkg_models[bkg]()
            # Combine the parameters and limits
            self.params = dict(_params1, **_params2, **_params3)
            self.limits = dict(_limits1, **_limits2, **_limits3)
            # Define the combined model function
            idx1 = len(_params1)
            idx2 = len(_params2) + idx1

            def model(x, *args):
                return _model1(x, args[:idx1]) + _model2(x, args[idx1:idx2]) + _model3(x, args[idx2:])

            self._model = model
            # Define the combined integral function

            def integral(x, *args):
                return _integral1(x, args[:idx1]) + _integral2(x, args[idx1:idx2]) + _integral3(x, args[idx2:])

            self._integral = integral
            # Define the combined derivative function
            if _jax_model1 is None or _jax_model2 is None or _jax_model3 is None:
                self._jax_model = None
            else:

                def jax_model(x, *args):
                    return _jax_model1(x, args[:idx1]) + _jax_model2(x, args[idx1:idx2]) + _jax_model3(x, args[idx2:])

                self._jax_model = jax_model

        # Keep previous parameters and limits if possible
        for par in _params_prev:
            if par in self.params:
                self.params[par] = self.m.values[par]

        # Update the cost function
        self.prepare_cost()

        # Update the Minuit object
        self.m = Minuit(self.cost, *list(self.params.values()), name=list(self.params.keys()))
        for par, lim in self.limits.items():
            self.m.limits[par] = lim

        # Remove the old fit widget
        self.layout.removeWidget(self.fit_widget)
        # Update the visualization
        self.fit_widget = make_widget(self.m, self.m._visualize(None), {}, False, False)
        self.layout.addWidget(self.fit_widget, 0, 0, 1, 3)

    def prepare_cost(self) -> None:
        # Dummy
        self.cost = None

    def _init_model_list(self) -> None:
        self.sig_models = {
            "Gaussian": self.gaussian,
            "Q-Gaussian": self.qgaussian,
            "Voigt": self.voigt,
            "Cruijff": self.cruijff,
            "CrystalBall": self.crystalball,
            "CrystalBallEx": self.crystalball_ex,
        }
        self.bkg_models = {
            "None": None,
            "Constant": self.constant,
            "Exponential": self.exponential,
            "Bernstein1d": partial(self.bernstein, 1),
            "Bernstein2d": partial(self.bernstein, 2),
            "Bernstein3d": partial(self.bernstein, 3),
            "Bernstein4d": partial(self.bernstein, 4),
        }

    def gaussian(self) -> Tuple[callable, callable, dict, dict]:
        dx = self.xr[1] - self.xr[0]
        params = {"s": self.sinit, "loc": 0.5 * (self.xr[0] + self.xr[1]),
                  "scale": 0.1 * dx}
        limits = {"s": (0, self.smax), "loc": self.xr,
                  "scale": (0.0001 * dx, 0.5 * dx)}

        def model(x, par):
            return par[0] * truncnorm.pdf(x, self.xr[0], self.xr[1], *par[1:])

        def integral(x, par):
            return par[0] * truncnorm.cdf(x, self.xr[0], self.xr[1], *par[1:])

        def jax_model(x, par):
            return par[0] * jnorm.pdf(x, *par[1:])

        return model, integral, jax_model, params, limits

    def qgaussian(self) -> Tuple[callable, callable, dict, dict]:
        dx = self.xr[1] - self.xr[0]
        params = {"s": self.sinit, "q": 2, "loc": 0.5 * (self.xr[0] + self.xr[1]),
                  "scale": 0.1 * dx}
        limits = {"s": (0, self.smax), "q": (1, 3), "loc": self.xr,
                  "scale": (0.0001 * dx, 0.5 * dx)}

        def model(x, par):
            if par[1] < 1:
                par[1] = 1
                warnings.warn("q cannot be smaller than 1. Setting q=1.")
            if par[1] > 3:
                par[1] = 3
                warnings.warn("q cannot be larger than 3. Setting q=3.")
            return par[0] * qgaussian.pdf(x, *par[1:])

        def integral(x, par):
            if par[1] < 1:
                par[1] = 1
                warnings.warn("q cannot be smaller than 1. Setting q=1.")
            if par[1] > 3:
                par[1] = 3
                warnings.warn("q cannot be larger than 3. Setting q=3.")
            return par[0] * qgaussian.cdf(x, *par[1:])

        return model, integral, None, params, limits

    def voigt(self) -> Tuple[callable, callable, dict, dict]:
        dx = self.xr[1] - self.xr[0]
        params = {"s": self.sinit, "gamma": 0.1 * dx,
                  "loc": 0.5 * (self.xr[0] + self.xr[1]), "scale": 0.1 * dx}
        limits = {"s": (0, self.smax), "gamma": (0.0001 * dx, 0.5 * dx),
                  "loc": self.xr, "scale": (0.0001 * dx, 0.5 * dx)}
        self._jit_numint_cdf()
        _x = np.linspace(self.xr[0], self.xr[1], 1000)

        def model(x, par):
            return par[0] * voigt.pdf(x, *par[1:])

        def integral(x, par):
            _cdf = self._numint_cdf(_x, voigt.pdf(_x, *par[1:]))
            return par[0] * np.interp(x, _x, _cdf)

        return model, integral, None, params, limits

    def cruijff(self) -> Tuple[callable, callable, dict, dict]:
        dx = self.xr[1] - self.xr[0]
        params = {"s": self.sinit, "beta_left": 0.1, "beta_right": 0.1,
                  "loc": 0.5 * (self.xr[0] + self.xr[1]),
                  "scale_left": 0.1 * dx, "scale_right": 0.1 * dx}
        limits = {"s": (0, self.smax), "beta_left": (0, 1), "beta_right": (0, 1),
                  "loc": self.xr, "scale_left": (0.0001 * dx, 0.5 * dx),
                  "scale_right": (0.0001 * dx, 0.5 * dx)}
        _x = np.linspace(self.xr[0], self.xr[1], 1000)
        self._jit_numint_cdf()

        def model(x, par):
            _cdf = self._numint_cdf(_x, cruijff.density(_x, *par[1:]))
            return par[0] / _cdf[-1] * cruijff.density(x, *par[1:])

        def integral(x, par):
            _cdf = self._numint_cdf(_x, cruijff.density(_x, *par[1:]))
            return par[0] / _cdf[-1] * np.interp(x, _x, _cdf)

        return model, integral, None, params, limits

    def crystalball(self) -> Tuple[callable, callable, dict, dict]:
        dx = self.xr[1] - self.xr[0]
        params = {"s": self.sinit, "beta": 1, "m": 2,
                  "loc": 0.5 * (self.xr[0] + self.xr[1]), "scale": 0.1 * dx}
        limits = {"s": (0, self.smax), "beta": (0, 5), "m": (1, 10),
                  "loc": self.xr, "scale": (0.0001 * dx, 0.5 * dx)}

        def model(x, par):
            return par[0] * crystalball.pdf(x, *par[1:])

        def integral(x, par):
            return par[0] * crystalball.cdf(x, *par[1:])

        return model, integral, None, params, limits

    def crystalball_ex(self) -> Tuple[callable, callable, dict, dict]:
        dx = self.xr[1] - self.xr[0]
        params = {"s": self.sinit, "beta_left": 1, "m_left": 2, "scale_left": 0.1 * dx,
                  "beta_right": 1, "m_right": 2, "scale_right": 0.1 * dx,
                  "loc": 0.5 * (self.xr[0] + self.xr[1])}
        limits = {"s": (0, self.smax), "beta_left": (0, 5), "m_left": (1, 10),
                  "scale_left": (0.0001 * dx, 0.5 * dx), "beta_right": (0, 5),
                  "m_right": (1, 10), "scale_right": (0.0001 * dx, 0.5 * dx),
                  "loc": self.xr}

        def model(x, par):
            return par[0] * crystalball_ex.pdf(x, *par[1:])

        def integral(x, par):
            return par[0] * crystalball_ex.cdf(x, *par[1:])

        return model, integral, None, params, limits

    def _jit_numint_cdf(self) -> None:
        if self._numint_cdf is not None:
            return
        # jit compile the numerical integration of the pdf

        @nb.njit(parallel=True, fastmath={"reassoc", "contract", "arcp"})
        def numint_cdf(_x, _pdf):
            y = np.empty_like(_x)
            for i in nb.prange(len(_x)):
                y[i] = np.trapz(_pdf[:i+1], x=_x[:i+1])
            return y

        self._numint_cdf = numint_cdf

    def constant(self) -> Tuple[callable, callable, dict, dict]:
        params = {"b": 1}
        limits = {"b": (0, None)}

        def model(x, par):
            return par[0] * uniform.pdf(x, self.xr[0], self.xr[1] - self.xr[0])

        def integral(x, par):
            return par[0] * uniform.cdf(x, self.xr[0], self.xr[1] - self.xr[0])

        return model, integral, None, params, limits

    def exponential(self) -> Tuple[callable, callable, dict, dict]:
        params = {"b": 1, "loc_expon": 0, "scale_expon": 1}
        limits = {"b": (0, None), "loc_expon": (-1, 0),
                  "scale_expon": (-100, 100)}

        def model(x, par):
            return par[0] * truncexpon.pdf(x, self.xr[0], self.xr[1], *par[1:])

        def integral(x, par):
            return par[0] * truncexpon.cdf(x, self.xr[0], self.xr[1], *par[1:])

        def jax_model(x, par):
            return par[0] * jexpon.pdf(x, *par[1:])

        return model, integral, jax_model, params, limits

    def bernstein(self, deg: int) -> Tuple[callable, callable, dict, dict]:
        params = {f"b_{i}{deg}": 1 for i in range(deg+1)}
        limits = {f"b_{i}{deg}": (0, None) for i in range(deg+1)}

        def model(x, args):
            return bernstein.density(x, args, self.xr[0], self.xr[1])

        def integral(x, args):
            return bernstein.integral(x, args, self.xr[0], self.xr[1])

        return model, integral, None, params, limits


class PhotonSpectrumFit(Fit2Signals1Background):
    def __init__(self,
        n: NDArray,
        xe: NDArray,
        sig: Union[str, Tuple[str, str]] = "Gaussian",
        bkg: str = "None",
        parent = None
    ) -> None:
        # Initialize fit data
        self.n = n
        self.xe = xe
        if len(n.shape) == 2:
            nsum = np.sum(n, axis=0)[0]
        else:
            nsum = np.sum(n)
        self.sinit = nsum * 0.75
        self.smax = nsum * 1.1

        super().__init__((xe[0], xe[-1]), sig=sig, bkg=bkg, parent=parent)

    def prepare_cost(self) -> None:
        self.cost = cost.ExtendedBinnedNLL(self.n, self.xe, self._integral)

    def _init_model_list(self) -> None:
        self.sig_models = {
            "Gaussian": self.gaussian,
            "Q-Gaussian": self.qgaussian,
            "Voigt": self.voigt,
            "Cruijff": self.cruijff,
            "CrystalBall": self.crystalball,
            "CrystalBallEx": self.crystalball_ex,
        }
        self.bkg_models = {
            "None": None,
            "Constant": self.constant,
            "Exponential": self.exponential,
            "Bernstein1d": partial(self.bernstein, 1),
            "Bernstein2d": partial(self.bernstein, 2),
            "Bernstein3d": partial(self.bernstein, 3),
            "Bernstein4d": partial(self.bernstein, 4),
        }


class PhotonExcitationFit(Fit2Signals1Background):
    def __init__(self,
        y: NDArray,
        yerr: NDArray,
        x: NDArray,
        xerr: NDArray,
        sig: Union[str, Tuple[str, str]] = "Gaussian",
        bkg: str = "Constant",
        constrain_dE: Tuple[float, float] = None,
        parent = None
    ) -> None:
        # Initialize fit data
        self.y = y
        self.yerr = yerr
        self.x = x
        self.xerr = xerr
        self.sinit = np.max(y) * (x[1] - x[0])
        self.smax = np.max(y) * (x[1] - x[0]) * 5
        self.constrain_dE = constrain_dE

        super().__init__((x[0], x[-1]), sig=sig, bkg=bkg, parent=parent)

    def prepare_fit(self) -> None:
        # Remember current parameters and limits
        _params_prev = self.params.copy()
        nconstraint = None
        if self.sig_comp2.currentText() == "None":
            self._model, self._derivative, self.params, self.limits = self.gaussian()
        else:
            self._model, self._derivative, self.params, self.limits = self.constrained_gaussians()
            if self.constrain_dE is not None:
                nconstraint = cost.NormalConstraint("loc2_loc1", *self.constrain_dE)

        # Keep previous parameters and limits if possible
        for par in _params_prev:
            if par in self.params:
                self.params[par] = self.m.values[par]

        jax.config.update("jax_enable_x64", True)

        @jax.jit
        def _cost(par):
            result = 0.0
            for xi, yi, dxi, dyi in zip(self.x, self.y, self.xerr, self.yerr):
                y_var = dyi**2 + (self._derivative(xi, *par) * dxi) ** 2
                result += (yi - self._model(xi, *par)) ** 2 / y_var
            return result

        class LeastSquaresXY(cost.LeastSquares):

            def _value(self, args: Sequence[float]) -> float:
                return _cost(args)

        # Update the cost function
        if nconstraint is None:
            self.cost = LeastSquaresXY(self.x, self.y, self.yerr, self._model)
        else:
            self.cost = LeastSquaresXY(self.x, self.y, self.yerr, self._model) + nconstraint

        # Update the Minuit object
        self.m = Minuit(self.cost, *list(self.params.values()), name=list(self.params.keys()))
        for par, lim in self.limits.items():
            self.m.limits[par] = lim

        def plot(args):
            plt.errorbar(self.x, self.y, yerr=self.yerr, xerr=self.xerr, fmt="ok")
            x = np.linspace(self.xr[0], self.xr[1], 1000)
            plt.plot(x, self._model(x, *args))

        # Remove the old fit widget
        self.layout.removeWidget(self.fit_widget)
        # Update the fit widget
        self.fit_widget = make_widget(self.m, plot, {}, False, False)
        self.layout.addWidget(self.fit_widget, 0, 0, 1, 3)

    def _init_model_list(self) -> None:
        self.sig_models = {
            "Gaussian": self.gaussian,
        }
        self.bkg_models = {
            "Constant": self.constant,
        }

    def gaussian(self) -> Tuple[callable, callable, dict, dict]:
        dx = self.xr[1] - self.xr[0]
        params = {
            "s1": self.sinit, "loc1": self.xr[0] + 0.5 * dx,
            "scale1": 0.1 * dx, "b": self.x[1] - self.x[0]
        }
        limits = {
            "s1": (0, self.smax), "loc1": self.xr,
            "scale1": (0.0001 * dx, 0.5 * dx), "b": (0, None)
        }

        def model(x, s1, loc1, scale1, b):
            return s1 * jnorm.pdf(x, loc1, scale1) + b

        def derivative(x, s1, loc1, scale1, b):
            return -s1 * jnorm.pdf(x, loc1, scale1) * (x - loc1) / scale1**2

        return model, derivative, params, limits

    def constrained_gaussians(self) -> Tuple[callable, callable, dict, dict]:
        dx = self.xr[1] - self.xr[0]
        params = {
            "s1": self.sinit, "s2": self.sinit,
            "loc1": self.xr[0] + 0.35 * dx,
            "loc2_loc1": 0.35 * dx,
            "scale1": 0.1 * dx, "b": self.x[1] - self.x[0]
        }
        limits = {
            "s1": (0, self.smax), "s2": (0, self.smax),
            "loc1": self.xr, "loc2_loc1": (0, dx),
            "scale1": (0.0001 * dx, 0.5 * dx), "b": (0, None)
        }

        def model(x, s1, s2, loc1, loc2_loc1, scale1, b):
            return (s1 * jnorm.pdf(x, loc1, scale1)
                    + s2 * jnorm.pdf(x, loc1 + loc2_loc1, scale1) + b)

        def derivative(x, s1, s2, loc1, loc2_loc1, scale1, b):
            return (-s1 * jnorm.pdf(x, loc1, scale1) * (x - loc1) / scale1**2
                    - s2 * jnorm.pdf(x, loc1 + loc2_loc1, scale1) * (x - loc1 - loc2_loc1) / scale1**2)

        return model, derivative, params, limits
