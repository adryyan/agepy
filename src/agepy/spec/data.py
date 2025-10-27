from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .instruments import Instrument
    from numpy.typing import NDArray


@dataclass(frozen=True)
class MetroValue:
    """Continuous data recorded by metro (#value).

    Parameters
    ----------
    values: np.ndarray, shape (N,)
        Recorded target pressures.

    """

    values: NDArray

    def mean(self, instr: Instrument | None = None) -> float:
        if instr is None:
            return np.mean(self.values)

        else:
            return instr.process(np.mean(self.values))

    def std(self, instr: Instrument | None = None) -> float:
        if instr is None:
            return np.std(self.values, ddof=1)

        else:
            return instr.process(np.std(self.values, ddof=1))

    def val(self, instr: Instrument | None = None) -> tuple[float, float]:
        mean = np.mean(self.values)
        std = np.std(self.values, ddof=1, mean=mean)

        if instr is not None:
            mean = instr.process(mean)
            std = instr.process(std)

        return mean, std
