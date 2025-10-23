from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "anodes",
    "PocoAnode",
    "WsaAnode",
    "DldAnodeXY",
    "DldAnodeUVW",
    "DldAnodeUV",
    "DldAnodeUW",
    "DldAnodeVW",
]


@dataclass(frozen=True)
class PositionAnode:
    """Generic anode as a parent class for all anodes.

    Parameters
    ----------
    angle: int or float
        Rotation of detector.
    scale: [float, float]
        scale = (scale_x, scale_y)
    offset: [float, float]
        offset = (offset_x, offset_y)

    """

    angle: int | float
    scale: tuple[float, float]
    offset: tuple[float, float]

    def __post_init__(self) -> None:
        t = np.deg2rad(self.angle, dtype=np.float64)
        st, ct = np.sin(t), np.cos(t)

        # Prepare rotation and scaling matrix
        transform = np.array(
            (
                (ct * self.scale[0], -st * self.scale[1]),
                (st * self.scale[0], ct * self.scale[1]),
            ),
            dtype=np.float64,
        )

        # Prepare the offset vector
        offset = np.array(self.offset, dtype=np.float64) + 0.5

        # Set as attributes
        object.__setattr__(self, "transform_matrix", transform)
        object.__setattr__(self, "offset_vector", offset)

    def process(self, rows: NDArray) -> NDArray:
        """Rotate, scale and shift the x, y coordinates.

        This is called by all anodes after conversion to
        x, y coordinates.

        """
        rows = rows[:, :2] - 0.5
        rows = rows.dot(self.transform_matrix) + self.offset_vector

        return rows


@dataclass(frozen=True)
class WsaAnode(PositionAnode):
    scale: tuple[float, float] = (3, 3)
    offset: tuple[float, float] = (0.8, 0.8)
    scale_a: float = 0.8
    scale_b: float = 0.65
    scale_c: float = 1.5
    offset_a: float = 0
    offset_b: float = 0
    offset_c: float = 0

    def process(self, rows):
        a = self.scale_a * rows[:, 0] + self.offset_a
        b = self.scale_b * rows[:, 1] + self.offset_b
        abc = a + b + self.scale_c * rows[:, 2] + self.offset_c

        pos = np.empty((a.shape[0], 2), dtype=float)
        pos[:, 0] = a / abc
        pos[:, 1] = b / abc

        return super().process(pos)


@dataclass(frozen=True)
class PocoAnode(PositionAnode):
    scale: tuple[float, float] = (5e-4, 5e-4)
    offset: tuple[float, float] = (0, 0)


@dataclass(frozen=True)
class DldAnode(PositionAnode):
    scale: tuple[float, float] = (1e-5, 1e-5)
    offset: tuple[float, float] = (0, 0)


@dataclass(frozen=True)
class DldAnodeXY(DldAnode):
    """DLD anode for the xy mode in metro and a Roentdek OpenFace
    quad delay-line detector.

    Parameters
    ----------
    angle: int or float
        Rotation of detector.

    """

    scale: tuple[float, float] = (5e-6, 5e-6)

    def process(self, rows: NDArray) -> NDArray:
        """Processes the raw data probably found in dld_rd#raw.

        Parameters
        ----------
        rows: np.ndarray, shape (N,4)
            Raw data from the detector.

        Returns
        -------
        pos: np.ndarray, shape (N,2)
            Processed data in form of xy values.

        """
        x = rows[:, 0] - rows[:, 1]
        y = rows[:, 2] - rows[:, 3]

        # TODO optimize calibration, the factor 1.06 is just by eye for now;
        # also, a cross correction should be done (software or hardware)
        ratio_xy = 1.06

        pos = np.empty((x.shape[0], 2), dtype=float)
        pos[:, 0] = x
        pos[:, 1] = y * ratio_xy

        return super().process(pos)


@dataclass(frozen=True)
class DldAnodeUVW(DldAnode):
    """Anode used with the uvw mode in metro and a Roentdek Hex
    delay-line detector.

    Parameters
    ----------
    angle: int or float
        Rotation of detector.

    """

    def process(self, rows: NDArray) -> NDArray:
        """Processes the raw data probably found in dld_rd#raw.

        Parameters
        ----------
        rows: np.ndarray, shape (N,6)
            Raw data from the detector.

        Returns
        -------
        pos: np.ndarray, shape (N,2)
            Processed data in form of xy values.

        """
        u = rows[:, 0] - rows[:, 1]
        v = rows[:, 2] - rows[:, 3]
        w = rows[:, 4] - rows[:, 5]

        pos = np.empty((u.shape[0], 2), dtype=float)
        pos[:, 0] = (2 * u - v - w) / 3
        pos[:, 1] = (v - w) / np.sqrt(3)

        return super().process(pos)


@dataclass(frozen=True)
class DldAnodeUV(DldAnode):
    def process(self, rows):
        u = rows[:, 0] - rows[:, 1]
        v = rows[:, 2] - rows[:, 3]

        pos = np.empty((u.shape[0], 2), dtype=float)
        pos[:, 0] = u
        pos[:, 1] = (u + 2 * v) / 1.7321

        return super().process(pos)


@dataclass(frozen=True)
class DldAnodeUW(DldAnode):
    def process(self, rows):
        if rows.shape[1] >= 6:
            u = rows[:, 0] - rows[:, 1]
            w = rows[:, 4] - rows[:, 5]

        else:
            u = rows[:, 0] - rows[:, 1]
            w = rows[:, 2] - rows[:, 3]

        pos = np.empty((u.shape[0], 2), dtype=float)
        pos[:, 0] = u
        pos[:, 1] = (u + 2 * w) / -1.7321

        return super().process(pos)


@dataclass(frozen=True)
class DldAnodeVW(DldAnode):
    def process(self, rows):
        if rows.shape[1] >= 6:
            v = rows[:, 2] - rows[:, 3]
            w = rows[:, 4] - rows[:, 5]

        else:
            v = rows[:, 0] - rows[:, 1]
            w = rows[:, 2] - rows[:, 3]

        pos = np.empty((v.shape[0], 2), dtype=float)
        pos[:, 0] = -v - w
        pos[:, 1] = (v - w) / 1.7321

        return super().process(pos)


anodes = {
    "poco": PocoAnode,
    "wsa": WsaAnode,
    "dld_xy": DldAnodeXY,
    "dld_uvw": DldAnodeUVW,
    "dld_uv": DldAnodeUV,
    "dld_uw": DldAnodeUW,
    "dld_vw": DldAnodeVW,
}
