from __future__ import annotations

from dataclasses import dataclass, field

from . import Q_

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from typing import Literal

__all__ = [
    "Keithley7510",
    "Keithley6485",
    "Keithley6514",
    "MKS270B",
    "MKS946",
]


@dataclass(frozen=True)
class Instrument:
    """Generic instrument as a parent class for all instruments."""

    unit: str
    data_unit: str

    def process(self, values: NDArray) -> NDArray:
        return Q_(values, self.data_unit).m_as(self.unit)


@dataclass(frozen=True)
class Keithley7510(Instrument):
    data_unit: str = field(default="V", init=False)


@dataclass(frozen=True)
class Keithley6485(Instrument):
    data_unit: str = field(default="A", init=False)


@dataclass(frozen=True)
class Keithley6514(Instrument):
    data_unit: str = field(default="A", init=False)


@dataclass(frozen=True)
class MKS270B(Instrument):
    """Pressures measured with the Baratron MKS270B and a Keithley
    as voltages.

    """

    unit: str
    setting: Literal["X1", "X.1", "X.01"]
    correction_factor: float = 1
    data_unit: str = field(default="torr", init=False)

    def process(self, values: NDArray) -> NDArray:
        if self.setting == "X1":
            values *= 0.1

        elif self.setting == "X.1":
            values *= 0.01

        elif self.setting == "X.01":
            values *= 0.001

        return super().process(values * self.correction_factor)


@dataclass(frozen=True)
class MKS946(Instrument):
    """Pressures measured with the Baratron MKS946."""

    unit: str
    setting: Literal["digital", "analog"]
    data_unit: str = field(default="torr", init=False)

    def process(self, values: NDArray) -> NDArray:
        if self.setting == "analog":
            values = 10 ** ((values - 7.2) / 0.6)

        return super().process(values)
