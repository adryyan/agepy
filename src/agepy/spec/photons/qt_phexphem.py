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
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.colors as colors
from agepy.qt import MainWindow
from agepy.qt.util import BlitManager
from agepy import ageplot
from agepy.spec.fit_models import (
    SumModel2d,
    FitModel2d,
    FitModel1d,
    Gaussian,
    Voigt,
    Constant,
    Exponential,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.backend_bases import MouseEvent
    from numpy.typing import NDArray
    from .energy_scan import EnergyScan


class ReferenceMarker:
    def __init__(
        self,
        parent: PhexPhemViewer,
        xy: tuple[float, float],
        pp: dict,
        width: float = 0.006,
        height: float = 0.015,
        x_fit: FitModel1d | None = None,
        y_fit: FitModel1d | None = None,
    ) -> None:
        self.parent = parent
        self.phexphem = pp
        self.width = width
        self.height = height
        self.pressed = False
        self.x_fit = None
        self.y_fit = None

        ini = f"X (v = 0, J = {pp['J']})"
        exc = f"{pp['Elp']} (v' = {pp['vp']}, J' = {pp['Jp']})"
        rlx = f"{pp['Elpp']} (v'' = {pp['vpp']}, J'' = {pp['Jpp']})"
        self.label = ini + " → " + exc + " → " + rlx

        # Draw ellipse
        self.marker = Ellipse(
            xy,
            width,
            height,
            color="red",
            fill=False,
            picker=True,
            animated=True,
        )
        self.parent.ax.add_patch(self.marker)

        # Connect mouse events
        cv = self.parent.canvas
        self.cid_p = cv.mpl_connect("button_press_event", self.on_press)
        self.cid_r = cv.mpl_connect("button_release_event", self.on_release)
        self.cid_m = cv.mpl_connect("motion_notify_event", self.on_motion)

        self.assign_fit(x_fit, y_fit)

    def get_center(self) -> tuple[float, float]:
        return self.marker.get_center()

    def assign_fit(self, x_fit, y_fit):
        if x_fit is not None and y_fit is not None:
            self.x_fit = x_fit
            self.y_fit = y_fit
            self.assigned = True
            self.marker.set(color="green")
            self.marker.set_center((x_fit.value("loc"), y_fit.value("loc")))
            self.parent.scan.set_assignment(x_fit, y_fit, **self.phexphem)

        else:
            self.x_fit = None
            self.y_fit = None
            self.assigned = False
            self.marker.set(color="red")
            self.parent.scan.set_assignment(None, None, **self.phexphem)

        self.parent.update()

    def remove(self) -> None:
        # Disconnect event callbacks
        cv = self.parent.canvas
        cv.mpl_disconnect(self.cid_p)
        cv.mpl_disconnect(self.cid_m)
        cv.mpl_disconnect(self.cid_r)

        # Remove the patch from the axes
        self.marker.remove()

    def on_press(self, event) -> None:
        contains, _ = self.marker.contains(event)
        if contains:
            self.pressed = True
            self.parent.toolbar.set_message(self.label)
            print(self.label)

    def on_motion(self, event) -> None:
        if self.pressed and event.inaxes == self.parent.ax:
            if self.assigned:
                self.assign_fit(None, None)
            self.marker.center = (event.xdata, event.ydata)
            self.parent.update()

    def on_release(self, event) -> None:
        self.pressed = False


class PhexPhemViewer(MainWindow):
    def __init__(
        self,
        scan: EnergyScan,
        reference: pd.DataFrame,
        phem_calib: tuple[float, float],
    ) -> None:
        self.scan = scan
        self.reference = reference
        self.phem_calib = phem_calib
        self.phex_calib = (0, 1)
        self.ref_markers = []

        # Prepare conversion of reference to detector coordinates
        b1 = 1 / phem_calib[1]
        b0 = -phem_calib[0] / phem_calib[1]
        self.phem_conv = (b0, b1)

        # Prepare the data
        self.map, self.err, self.xe, self.ye = scan.phexphem(
            bkg=False,
            calib=False,
            mc_errors=False,
        )

        # Set up a meshgrid for plotting
        self.xm, self.ym = np.meshgrid(self.xe, self.ye)

        # Set up the main window
        super().__init__(title="PhexPhem Viewer", width=1000, height=760)

        self.add_plot(width=960, height=720)
        self.add_toolbar()

        # Add button for displaying the reference
        self.ref_action = self.add_action(
            self.plot_reference, text="Show Reference", checkable=True
        )

        # Add the fit action
        self.fit_action, self.selector = self.add_rect_selector(
            self.ax, self.on_select, text="Fit Assignments"
        )

        # Only enable the fit action when the reference is shown
        self.fit_action.setEnabled(False)

        # Add button for displaying the assignments
        self.asn_action = self.add_action(
            self.plot_assignments, text="Show Assignments", checkable=True
        )

        # Get the default Qt window background color
        palette = QtWidgets.QApplication.palette()
        bg = palette.color(QtGui.QPalette.Window)
        rgb = (bg.red() / 255.0, bg.green() / 255.0, bg.blue() / 255.0)

        self.fig.set_facecolor(rgb)
        self.fig.set_edgecolor(rgb)

        self.bm = None

        with ageplot.context(["age", "qt"]):
            # Remove grid from the map
            self.ax.grid(False)

            self.ax.tick_params(axis="both", which="both", direction="out")

            # Limit x axis
            self.ax.set_xlim(11, 18)
            # Flip y axis
            self.ax.set_ylim(0.9, 0.1)

            # Set labels
            self.ax.set_xlabel("Exciting Photon Energy [eV]")
            self.ax.set_ylabel("Detector Position [arb. u.]")

        self.plot()

    def plot(self) -> None:
        vmax = self.map.max()

        def _forward(x):
            return 1 - np.exp(-20 * x / vmax)

        def _inverse(x):
            return -np.log(1 - x) * vmax / 20

        with ageplot.context(["age", "qt"]):
            # Plot the map
            self.ax.pcolormesh(
                self.xm,
                self.ym,
                self.map.T,
                cmap="viridis",
                rasterized=True,
                norm=colors.FuncNorm(
                    (_forward, _inverse), vmin=0, vmax=vmax * 0.7, clip=True
                ),
            )

            self.canvas.draw_idle()

    def plot_reference(self):
        # Remove markers and disable fit action
        if not self.ref_action.isChecked():
            self.fit_action.setChecked(False)
            self.fit_action.setEnabled(False)

            if self.bm is not None:
                self.bm.close()

            for mk in self.ref_markers:
                mk.remove()

            self.ref_markers = []
            self.bm = None

            self.canvas.draw_idle()
            return

        # Uncheck assignment action
        if self.asn_action.isChecked():
            self.asn_action.setChecked(False)

        # Select phexphem transitions in the current limits
        xlim = self.ax.get_xlim()  # noqa F841
        ylim = self.ax.get_ylim()  # noqa F841

        ref = self.reference.query("exc_energy > @xlim[0]").query(
            "exc_energy < @xlim[1]"
        )

        # Draw a marker for each transition
        with ageplot.context(["age", "qt"]):
            for row in ref.itertuples():
                x = row.exc_energy
                y = row.emi_energy * self.phem_conv[1] + self.phem_conv[0]

                qn = {
                    "J": row.J,
                    "Elp": row.Elp,
                    "vp": row.vp,
                    "Jp": row.Jp,
                    "Elpp": row.Elpp,
                    "vpp": row.vpp,
                    "Jpp": row.Jpp,
                }

                x_fit, y_fit = self.scan.get_assignment(**qn)

                self.ref_markers.append(
                    ReferenceMarker(self, (x, y), qn, x_fit=x_fit, y_fit=y_fit)
                )

            self.bm = BlitManager(
                self.canvas, [rm.marker for rm in self.ref_markers]
            )

            self.update()
            self.update()

        self.fit_action.setEnabled(True)

    def plot_assignments(self):
        if not self.asn_action.isChecked():
            pass

        if self.ref_action.isChecked():
            self.ref_action.setChecked(False)

    def update(self):
        if self.bm is None:
            return

        self.bm.update()

    def on_select(self, eclick: MouseEvent, erelease: MouseEvent):
        # Get the selection
        xr = np.sort((eclick.xdata, erelease.xdata))
        yr = np.sort((eclick.ydata, erelease.ydata))

        # Select the data in the range
        ix = np.where((self.xe[:-1] >= xr[0]) & (self.xe[1:] <= xr[1]))[0]
        iy = np.where((self.ye[:-1] >= yr[0]) & (self.ye[1:] <= yr[1]))[0]

        hist = self.map[np.ix_(ix, iy)].copy()
        err = self.err[np.ix_(ix, iy)].copy()

        n = np.stack((hist, err * 2), axis=-1)
        xe = self.xe[ix[0] : ix[-1] + 2]
        ye = self.ye[iy[0] : iy[-1] + 2]

        # Collect the assignments in the range
        assignments = []
        for mk in self.ref_markers:
            x, y = mk.get_center()
            if x > xr[0] and x < xr[1] and y > yr[0] and y < yr[1]:
                assignments.append(mk)

        if len(assignments) == 0:
            self.toolbar.set_message("No assignments in selection.")
            return

        debug_fit = InteractiveFit(self, n, xe, ye, assignments)
        debug_fit.exec()

        plt.close("all")


class InteractiveFit(QtWidgets.QDialog):
    sig_models = {
        "Gaussian": Gaussian,
        "Voigt": Voigt,
    }

    bkg_models = {
        "Constant": Constant,
        "Exponential": Exponential,
    }

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        n: NDArray,
        xe: NDArray,
        ye: NDArray,
        assignments: list[ReferenceMarker],
        x_sig: str = "Gaussian",
        y_sig: str = "Voigt",
        x_bkg: str = "Constant",
        y_bkg: str = "Constant",
    ) -> None:
        self.n, self.xe, self.ye = n, xe, ye
        self.xr = (xe[0], xe[-1])
        self.yr = (ye[0], ye[-1])
        self.assignments = assignments
        self.n_sig = len(assignments)
        self.fit = None

        # Initialize the parent class
        super().__init__(parent)
        self.setWindowTitle("PhexPhem Fit")
        self.resize(1280, 720)
        self.setMaximumSize(1280, 720)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.setFont(font)

        self.layout = QtWidgets.QGridLayout(self)

        # Create dummy plot
        self.fit_widget = QtWidgets.QWidget()
        self.layout.addWidget(self.fit_widget, 0, 0, 1, 3)

        # Create size policy
        size_policy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.MinimumExpanding,
            QtWidgets.QSizePolicy.Policy.MinimumExpanding,
        )

        # Create signal model selection widget
        self.sig = {}
        for i, mk in enumerate(assignments):
            x_loc, y_loc = mk.get_center()
            loc_str = f" @ x = {x_loc:.3f}, y = {y_loc:.3f}"
            group = QtWidgets.QGroupBox(f"{i}: " + mk.label + loc_str)
            group.setSizePolicy(size_policy)
            layout = QtWidgets.QHBoxLayout(group)

            # Create ComboBox for the x signal component
            x_cbox = QtWidgets.QComboBox()
            x_cbox.addItems(self.sig_models.keys())
            x_cbox.setCurrentIndex(list(self.sig_models.keys()).index(x_sig))
            x_cbox.currentIndexChanged.connect(self.prepare_fit)
            layout.addWidget(x_cbox)

            # Create ComboBox for the y signal component
            y_cbox = QtWidgets.QComboBox()
            y_cbox.addItems(self.sig_models.keys())
            y_cbox.setCurrentIndex(list(self.sig_models.keys()).index(y_sig))
            y_cbox.currentIndexChanged.connect(self.prepare_fit)
            layout.addWidget(y_cbox)

            self.layout.addWidget(group, i + 1, 0, 1, 3)

            self.sig[i] = {
                "group": group,
                "layout": layout,
                "x": x_cbox,
                "x_fit": None,
                "y": y_cbox,
                "y_fit": None,
                "fit": None,
            }

        # Create background model selection widget
        group = QtWidgets.QGroupBox("Background Model")
        group.setSizePolicy(size_policy)
        layout = QtWidgets.QHBoxLayout(group)

        x_cbox = QtWidgets.QComboBox()
        x_cbox.addItems(self.bkg_models.keys())
        x_cbox.setCurrentIndex(list(self.bkg_models.keys()).index(x_bkg))
        x_cbox.currentIndexChanged.connect(self.prepare_fit)
        layout.addWidget(x_cbox)

        y_cbox = QtWidgets.QComboBox()
        y_cbox.addItems(self.bkg_models.keys())
        y_cbox.setCurrentIndex(list(self.bkg_models.keys()).index(y_bkg))
        y_cbox.currentIndexChanged.connect(self.prepare_fit)
        layout.addWidget(y_cbox)

        self.layout.addWidget(group, i + 2, 0, 1, 2)

        self.bkg = {
            "group": group,
            "layout": layout,
            "x": x_cbox,
            "x_fit": None,
            "y": y_cbox,
            "y_fit": None,
            "fit": None,
        }

        # Create the button box
        size_policy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Minimum,
            QtWidgets.QSizePolicy.Policy.MinimumExpanding,
        )
        self.button_group = QtWidgets.QGroupBox("Add Fit Result")
        self.button_group.setSizePolicy(size_policy)
        self.button_layout = QtWidgets.QHBoxLayout(self.button_group)
        self.button_box = QtWidgets.QDialogButtonBox(parent=self)
        self.button_box.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.button_box.setStandardButtons(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
            | QtWidgets.QDialogButtonBox.StandardButton.Ok
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        QtCore.QMetaObject.connectSlotsByName(self)
        self.button_layout.addWidget(self.button_box)
        self.layout.addWidget(
            self.button_group,
            i + 2,
            2,
            1,
            1,
            alignment=QtCore.Qt.AlignmentFlag.AlignLeft,
        )

        # Set the column stretch factors
        self.layout.setColumnStretch(0, 1)
        self.layout.setColumnStretch(1, 1)
        self.layout.setColumnStretch(2, 0)

        # Create the initial fit widget
        self.prepare_fit()

    def update_fit_widget(self, widget: QtWidgets.QWidget) -> None:
        # Remove the old fit widget
        self.layout.removeWidget(widget)
        # Set the new fit widget
        self.fit_widget = widget
        self.layout.addWidget(self.fit_widget, 0, 0, 1, 3)

    def prepare_fit(self) -> None:
        if self.fit is None:
            self.fit = SumModel2d()

        for i in range(self.n_sig):
            x_name = self.sig[i]["x"].currentText()
            y_name = self.sig[i]["y"].currentText()

            if self.sig[i]["fit"] is not None:
                x_prev = self.sig[i]["x_fit"].name
                y_prev = self.sig[i]["y_fit"].name
                if x_name == x_prev and y_name == y_prev:
                    continue

            x_fit = self.sig_models[x_name](self.xr)
            self.sig[i]["x_fit"] = x_fit
            y_fit = self.sig_models[y_name](self.yr)
            self.sig[i]["y_fit"] = y_fit

            x_loc, y_loc = self.assignments[i].get_center()
            x_w = self.assignments[i].width
            y_w = self.assignments[i].height
            x_fit.set_value("loc", x_loc)
            x_fit.lim["loc"] = (x_loc - x_w, x_loc + x_w)
            x_fit.set_value("scale", x_w * 0.1)
            x_fit.lim["scale"] = (x_w * 0.001, x_w)
            y_fit.set_value("loc", y_loc)
            y_fit.lim["loc"] = (y_loc - y_w, y_loc + y_w)
            y_fit.set_value("scale", y_w * 0.1)
            y_fit.lim["scale"] = (y_w * 0.001, y_w)

            self.sig[i]["fit"] = FitModel2d(x_fit, y_fit)
            self.fit.add_model(self.sig[i]["fit"], idx=i)

        x_name = self.bkg["x"].currentText()
        y_name = self.bkg["y"].currentText()

        bkg_changed = True
        if self.bkg["fit"] is not None:
            x_prev = self.bkg["x_fit"].name
            y_prev = self.bkg["y_fit"].name
            bkg_changed = (x_name != x_prev) or (y_name != y_prev)

        if self.bkg["fit"] is None or bkg_changed:
            x_fit = self.bkg_models[x_name](self.xr)
            self.bkg["x_fit"] = x_fit
            y_fit = self.bkg_models[y_name](self.yr)
            self.bkg["y_fit"] = y_fit

            self.bkg["fit"] = FitModel2d(x_fit, y_fit)
            self.fit.add_model(self.bkg["fit"], idx=i + 1)

        # Update the cost function
        c = cost.ExtendedBinnedNLL(
            self.n, (self.xe, self.ye), self.fit.integral
        )

        # Update the Minuit object
        self.m = Minuit(c, *list(self.fit.val), name=self.fit.par)

        # Set the limits
        for par, lim in self.fit.lim.items():
            self.m.limits[par] = lim

        # Update the visualization
        fit_widget = make_widget(
            self.m, self.m._visualize(None), {}, False, False
        )

        # Perform the fit
        fit_widget.fit_button.click()

        # Update the layout
        self.update_fit_widget(fit_widget)

    def accept(self) -> None:
        if not self.m.valid:
            super().reject()

        self.fit.val = np.array(self.m.values)
        self.fit.err = np.array(self.m.errors)

        for i in range(self.n_sig):
            self.assignments[i].assign_fit(
                self.sig[i]["x_fit"], self.sig[i]["y_fit"]
            )

        super().accept()
