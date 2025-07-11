from __future__ import annotations

try:
    from PySide6 import QtWidgets, QtCore, QtGui

except ImportError as e:
    errmsg = "PySide6 required for interactive fitting."
    raise ImportError(errmsg) from e

try:
    from iminuit import Minuit, cost
    from iminuit.qtwidget import make_widget

except ImportError as e:
    errmsg = "iminuit required for fitting."
    raise ImportError(errmsg) from e

import numpy as np
from jacobi import propagate
import pandas as pd
import matplotlib.pyplot as plt

from agepy.interactive import MainWindow
from ._assignment_dialog import AssignmentDialog
from ._interactive_fit import Gaussian, Constant
from agepy import ageplot

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.backend_bases import MouseEvent
    from numpy.typing import NDArray, ArrayLike
    from .energy_scan import EnergyScan
    from ._interactive_fit import FitModel


class PhexPhemViewer(MainWindow):
    def __init__(
        self,
        scan: EnergyScan,
        reference: pd.DataFrame,
    ) -> None:
        # Set the scan and reference data
        self.scan = scan
        self.reference = reference

        # Initialize the parent class
        super().__init__()

        # Set up the main window
        self.add_plot()
        self.add_toolbar()

        # Prepare the data
        self.map, self.err, self.xe, self.ye = scan.phexphem(
            calib=False,
        )
