from __future__ import annotations
from typing import Union, Tuple
import numpy as np


# Classes implementing the various types of anodes

class PositionAnode(object):
    """Generic anode as a parent class for all anodes.

    Parameters
    ----------
    angle: int or float
        Rotation of detector.
    scale: tuple
        scale = (scale_x, scale_y)
    offset: tuple
        offset = (offset_x, offset_y)

    """

    def __init__(self, angle: Union[int, float], scale: Tuple[float, float],
                 offset: Tuple[float, float]) -> None:
        self.transform = np.zeros((2, 2), dtype=float)
        self.offset = np.zeros((2,), dtype=float)
        pre_v, post_v = angle * np.pi/180, 0 * np.pi/180
        pre_sin, post_sin = np.sin(pre_v), np.sin(post_v)
        pre_cos, post_cos = np.cos(pre_v), np.cos(post_v)
        # First, rotate by the pre matrix, then scale and
        # afterwards rotate again by the post matrix.
        # The components are the product of these three matrices in
        # order.
        self.transform[0,0] = (pre_cos * post_cos * (+scale[0])
                               + pre_sin * post_sin * (-scale[1]))
        self.transform[1,0] = (pre_sin * post_cos * (+scale[0])
                               + pre_cos * post_sin * (+scale[1]))
        self.transform[0,1] = (pre_cos * post_sin * (-scale[0])
                               + pre_sin * post_cos * (-scale[1]))
        self.transform[1,1] = (pre_sin * post_sin * (-scale[0])
                               + pre_cos * post_cos * (+scale[1]))
        # +0.5 for the offset introduced by centering the image on
        # the origin.
        self.offset[0] = offset[0] + 0.5
        self.offset[1] = offset[1] + 0.5        

    def process(self, rows):
        """Needs documentation.

        """
        rows = rows[:, :2] - 0.5
        rows = rows.dot(self.transform) + self.offset

        return rows


class WsaAnode(PositionAnode):

    def __init__(self, angle, scale=(3, 3), offset=(0.8, 0.8),
                 scale_a=0.8, scale_b=0.65, scale_c=1.5,
                 offset_a=0, offset_b=0, offset_c=0):
        super().__init__(angle, scale, offset)

        self.scale_a = scale_a
        self.scale_b = scale_b
        self.scale_c = scale_c

        self.offset_a = offset_a
        self.offset_b = offset_b
        self.offset_c = offset_c

    def process(self, rows):
        a = self.scale_a * rows[:, 0] + self.offset_a
        b = self.scale_b * rows[:, 1] + self.offset_b
        abc = a + b + self.scale_c * rows[:, 2] + self.offset_c

        pos = np.empty((a.shape[0], 2), dtype=float)
        pos[:, 0] = a / abc
        pos[:, 1] = b / abc

        return super().process(pos)


class PocoAnode(PositionAnode):

    def __init__(self, angle):
        super().__init__(angle, (0.0005, 0.0005), (0, 0))


class DldAnode(PositionAnode):

    def __init__(self, angle):
        super().__init__(angle, (1e-05, 1e-05), (0, 0))


class DldAnodeXY(DldAnode):
    """DLD anode for the xy mode in metro and a Roentdek OpenFace
    quad delay-line detector.

    Parameters
    ----------
    angle: int or float
        Rotation of detector.

    """

    def process(self, rows: np.ndarray) -> np.ndarray:
        """Processes the raw data probably found in dld_rd#raw.

        Parameters
        ----------
        rows: np.ndarray
            Raw data from the detector.

        Returns
        -------
        np.ndarray
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


class DldAnodeUVW(DldAnode):
    """Anode used with the uvw mode in metro and a Roentdek Hex
    delay-line detector.

    Parameters
    ----------
    angle: int or float
        Rotation of detector.

    """

    def process(self, rows: np.ndarray) -> np.ndarray:
        """Processes the raw data probably found in dld_rd#raw.

        Parameters
        ----------
        rows: np.ndarray
            Raw data from the detector.

        Returns
        -------
        np.ndarray
            Processed data in form of xy values.
        
        """
        u = rows[:, 0] - rows[:, 1]
        v = rows[:, 2] - rows[:, 3]
        w = rows[:, 4] - rows[:, 5]

        pos = np.empty((u.shape[0], 2), dtype=float)
        pos[:, 0] = (2*u - v - w) / 3
        pos[:, 1] = (v - w) / np.sqrt(3)

        return super().process(pos)


class DldAnodeUV(DldAnode):

    def process(self, rows):
        u = rows[:, 0] - rows[:, 1]
        v = rows[:, 2] - rows[:, 3]

        pos = np.empty((u.shape[0], 2), dtype=float)
        pos[:, 0] = u
        pos[:, 1] = (u + 2*v) / 1.7321

        return super().process(pos)


class DldAnodeUW(DldAnode):

    def process(self, rows):
        u = rows[:, 0] - rows[:, 1]
        w = rows[:, 4] - rows[:, 5]

        pos = np.empty((u.shape[0], 2), dtype=float)
        pos[:, 0] = u
        pos[:, 1] = (u + 2*w) / -1.7321

        return super().process(pos)


class DldAnodeVW(DldAnode):

    def process(self, rows):
        v = rows[:, 2] - rows[:, 3]
        w = rows[:, 4] - rows[:, 5]

        pos = np.empty((v.shape[0], 2), dtype=float)
        pos[:, 0] = - v - w
        pos[:, 1] = (v - w) / 1.7321

        return super().process(pos)


class Old_PositionAnode(PositionAnode):

    def __init__(self, angle, scale, offset):
        self.transform = np.zeros((2,2), dtype=float)
        self.offset = np.zeros((2,), dtype=float)

        self.offset[0] = offset[0] + 0.5/scale[0]
        self.offset[1] = offset[1] + 0.5/scale[1]

        v = angle * 3.14/180
        sin, cos = np.sin(v), np.cos(v)

        self.transform[0,0] = cos * scale[0]
        self.transform[1,0] = sin * scale[0]
        self.transform[0,1] = -sin * scale[1]
        self.transform[1,1] = cos * scale[1]

    def process(self, rows):
        rows = rows - self.offset
        rows = rows.dot(self.transform) + 0.5

        return rows


class Old_WsaAnode(Old_PositionAnode):

    def __init__(self, angle, scale=(2.65, 2.70), offset=(0.04, 0.06),
                 scale_a=0.80, scale_b=0.65, scale_c=1.50,
                 offset_a=0, offset_b=0, offset_c=0):
        super().__init__(angle, scale, offset)

        self.scale_a = scale_a
        self.scale_b = scale_b
        self.scale_c = scale_c

        self.offset_a = offset_a
        self.offset_b = offset_b
        self.offset_c = offset_c

    def process(self, rows):
        a = self.scale_a * rows[:, 0] + self.offset_a
        b = self.scale_b * rows[:, 1] + self.offset_b
        abc = a + b + self.scale_c * rows[:, 2] + self.offset_c

        pos = np.empty((a.shape[0], 2), dtype=float)
        pos[:, 0] = a / abc
        pos[:, 1] = b / abc

        return super().process(pos)


class Old_DldAnode(Old_PositionAnode):

    def __init__(self, angle):
        super().__init__(angle, (0.0005, 0.0005), (-1024, -1024))

    def process(self, rows):
        u = rows[:, 0] - rows[:, 1]
        v = rows[:, 2] - rows[:, 3]
        w = rows[:, 4] - rows[:, 5]

        pos = np.empty((u.shape[0], 2), dtype=float)
        pos[:, 0] = (2*u - v - w) / 3
        pos[:, 1] = (v - w) / 1.7321

        return super().process(pos)


available_anodes = {
    'poco': PocoAnode,
    'wsa': WsaAnode,
    'dld_xy': DldAnodeXY,
    'dld_uvw': DldAnodeUVW,
    'dld_uv': DldAnodeUV,
    'dld_uw': DldAnodeUW,
    'dld_vw': DldAnodeVW,
    'old_wsa': Old_WsaAnode,
    'old_dld': Old_DldAnode
}
