# Import PySide6 / PyQt6 modules
from agepy.interactive.util import import_qt_binding
qt_binding, QtWidgets, QtCore, QtGui = import_qt_binding()


class Ui_PhemDialog(object):
    def setupUi(self, PhemDialog):
        PhemDialog.setObjectName("PhemDialog")
        PhemDialog.resize(400, 137)
        font = QtGui.QFont()
        font.setPointSize(14)
        PhemDialog.setFont(font)
        self.grid_layout = QtWidgets.QGridLayout(PhemDialog)
        self.grid_layout.setObjectName("grid_layout")
        self.button_box = QtWidgets.QDialogButtonBox(parent=PhemDialog)
        self.button_box.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.button_box.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.Cancel|QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.button_box.setObjectName("button_box")
        self.grid_layout.addWidget(self.button_box, 1, 0, 1, 2)
        self.Jpp_group = QtWidgets.QGroupBox(parent=PhemDialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Jpp_group.sizePolicy().hasHeightForWidth())
        self.Jpp_group.setSizePolicy(sizePolicy)
        self.Jpp_group.setObjectName("Jpp_group")
        self.Jpp_layout = QtWidgets.QHBoxLayout(self.Jpp_group)
        self.Jpp_layout.setObjectName("Jpp_layout")
        self.Jpp_input = QtWidgets.QSpinBox(parent=self.Jpp_group)
        self.Jpp_input.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.Jpp_input.setObjectName("Jpp_input")
        self.Jpp_layout.addWidget(self.Jpp_input)
        self.grid_layout.addWidget(self.Jpp_group, 0, 1, 1, 1)
        self.vpp_group = QtWidgets.QGroupBox(parent=PhemDialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.vpp_group.sizePolicy().hasHeightForWidth())
        self.vpp_group.setSizePolicy(sizePolicy)
        self.vpp_group.setObjectName("vpp_group")
        self.vpp_layout = QtWidgets.QHBoxLayout(self.vpp_group)
        self.vpp_layout.setObjectName("vpp_layout")
        self.vpp_input = QtWidgets.QSpinBox(parent=self.vpp_group)
        self.vpp_input.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.vpp_input.setObjectName("vpp_input")
        self.vpp_layout.addWidget(self.vpp_input)
        self.grid_layout.addWidget(self.vpp_group, 0, 0, 1, 1)

        self.retranslateUi(PhemDialog)
        self.button_box.accepted.connect(PhemDialog.accept) # type: ignore
        self.button_box.rejected.connect(PhemDialog.reject) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(PhemDialog)

    def retranslateUi(self, PhemDialog):
        _translate = QtCore.QCoreApplication.translate
        PhemDialog.setWindowTitle(_translate("PhemDialog", "Assignment Dialog"))
        self.Jpp_group.setTitle(_translate("PhemDialog", "J\'\'"))
        self.vpp_group.setTitle(_translate("PhemDialog", "v\'\'"))
