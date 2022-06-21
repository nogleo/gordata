# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(773, 445)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMaximumSize(QtCore.QSize(2500, 2000))
        MainWindow.setBaseSize(QtCore.QSize(700, 450))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        self.tabWidget.setObjectName("tabWidget")
        self.tab_daq = QtWidgets.QWidget()
        self.tab_daq.setObjectName("tab_daq")
        self.gridLayout = QtWidgets.QGridLayout(self.tab_daq)
        self.gridLayout.setObjectName("gridLayout")
        self.stopbutton = QtWidgets.QPushButton(self.tab_daq)
        self.stopbutton.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.stopbutton.sizePolicy().hasHeightForWidth())
        self.stopbutton.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(40)
        self.stopbutton.setFont(font)
        self.stopbutton.setCheckable(False)
        self.stopbutton.setObjectName("stopbutton")
        self.gridLayout.addWidget(self.stopbutton, 3, 0, 1, 1)
        self.horizontalSlider = QtWidgets.QSlider(self.tab_daq)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.horizontalSlider.sizePolicy().hasHeightForWidth())
        self.horizontalSlider.setSizePolicy(sizePolicy)
        self.horizontalSlider.setMinimumSize(QtCore.QSize(100, 55))
        self.horizontalSlider.setSizeIncrement(QtCore.QSize(0, 0))
        self.horizontalSlider.setBaseSize(QtCore.QSize(10, 10))
        font = QtGui.QFont()
        font.setPointSize(30)
        self.horizontalSlider.setFont(font)
        self.horizontalSlider.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.horizontalSlider.setMaximum(120)
        self.horizontalSlider.setPageStep(5)
        self.horizontalSlider.setSliderPosition(0)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setInvertedAppearance(False)
        self.horizontalSlider.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        self.horizontalSlider.setTickInterval(5)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.gridLayout.addWidget(self.horizontalSlider, 3, 3, 1, 3)
        self.startbutton = QtWidgets.QPushButton(self.tab_daq)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.startbutton.sizePolicy().hasHeightForWidth())
        self.startbutton.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(40)
        self.startbutton.setFont(font)
        self.startbutton.setAutoFillBackground(False)
        self.startbutton.setObjectName("startbutton")
        self.gridLayout.addWidget(self.startbutton, 0, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.tab_daq)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(60)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 3, 1, 2)
        self.label = QtWidgets.QLabel(self.tab_daq)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(60)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 5, 1, 1)
        self.tabWidget.addTab(self.tab_daq, "")
        self.tab_plot = QtWidgets.QWidget()
        self.tab_plot.setObjectName("tab_plot")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.tab_plot)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.vLayout_plot = QtWidgets.QVBoxLayout()
        self.vLayout_plot.setObjectName("vLayout_plot")
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.vLayout_plot.addItem(spacerItem)
        self.gridLayout_2.addLayout(self.vLayout_plot, 3, 0, 1, 3)
        self.hLayout_plot = QtWidgets.QHBoxLayout()
        self.hLayout_plot.setObjectName("hLayout_plot")
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.hLayout_plot.addItem(spacerItem1)
        self.gridLayout_2.addLayout(self.hLayout_plot, 1, 1, 1, 2)
        self.openbttn = QtWidgets.QPushButton(self.tab_plot)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.openbttn.sizePolicy().hasHeightForWidth())
        self.openbttn.setSizePolicy(sizePolicy)
        self.openbttn.setMinimumSize(QtCore.QSize(50, 30))
        self.openbttn.setMaximumSize(QtCore.QSize(50, 30))
        self.openbttn.setObjectName("openbttn")
        self.gridLayout_2.addWidget(self.openbttn, 1, 0, 1, 1)
        self.tabWidget.addTab(self.tab_plot, "")
        self.tab_TF = QtWidgets.QWidget()
        self.tab_TF.setObjectName("tab_TF")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.tab_TF)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.loadbttn = QtWidgets.QPushButton(self.tab_TF)
        self.loadbttn.setMinimumSize(QtCore.QSize(50, 30))
        self.loadbttn.setMaximumSize(QtCore.QSize(50, 30))
        self.loadbttn.setObjectName("loadbttn")
        self.gridLayout_5.addWidget(self.loadbttn, 0, 0, 1, 1)
        self.hLayout_TF = QtWidgets.QHBoxLayout()
        self.hLayout_TF.setObjectName("hLayout_TF")
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.hLayout_TF.addItem(spacerItem2)
        self.gridLayout_5.addLayout(self.hLayout_TF, 0, 2, 1, 1)
        self.combo_TF = QtWidgets.QComboBox(self.tab_TF)
        self.combo_TF.setMinimumSize(QtCore.QSize(70, 30))
        self.combo_TF.setMaximumSize(QtCore.QSize(100, 30))
        self.combo_TF.setObjectName("combo_TF")
        self.gridLayout_5.addWidget(self.combo_TF, 0, 1, 1, 1)
        self.vLayout_TF = QtWidgets.QVBoxLayout()
        self.vLayout_TF.setObjectName("vLayout_TF")
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.vLayout_TF.addItem(spacerItem3)
        self.gridLayout_5.addLayout(self.vLayout_TF, 1, 0, 1, 3)
        self.tabWidget.addTab(self.tab_TF, "")
        self.tab_calib = QtWidgets.QWidget()
        self.tab_calib.setEnabled(True)
        self.tab_calib.setObjectName("tab_calib")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.tab_calib)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.calibutton = QtWidgets.QPushButton(self.tab_calib)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.calibutton.setFont(font)
        self.calibutton.setObjectName("calibutton")
        self.gridLayout_3.addWidget(self.calibutton, 0, 5, 1, 1)
        self.comboBox = QtWidgets.QComboBox(self.tab_calib)
        self.comboBox.setObjectName("comboBox")
        self.gridLayout_3.addWidget(self.comboBox, 0, 3, 1, 1)
        self.initbttn = QtWidgets.QPushButton(self.tab_calib)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.initbttn.setFont(font)
        self.initbttn.setObjectName("initbttn")
        self.gridLayout_3.addWidget(self.initbttn, 0, 2, 1, 1)
        self.linkSensor = QtWidgets.QPushButton(self.tab_calib)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.linkSensor.setFont(font)
        self.linkSensor.setObjectName("linkSensor")
        self.gridLayout_3.addWidget(self.linkSensor, 0, 4, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_3.addItem(spacerItem4, 1, 3, 1, 1)
        self.tabWidget.addTab(self.tab_calib, "")
        self.verticalLayout.addWidget(self.tabWidget)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        self.horizontalSlider.valueChanged['int'].connect(self.label.setNum)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "DATANOG"))
        self.stopbutton.setText(_translate("MainWindow", "Stop"))
        self.startbutton.setToolTip(_translate("MainWindow", "\'Start collecting data\'"))
        self.startbutton.setText(_translate("MainWindow", "Start"))
        self.label_2.setText(_translate("MainWindow", "Duration: "))
        self.label.setText(_translate("MainWindow", "0"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_daq), _translate("MainWindow", "DAQ"))
        self.openbttn.setText(_translate("MainWindow", "Load"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_plot), _translate("MainWindow", "Process"))
        self.loadbttn.setText(_translate("MainWindow", "Load"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_TF), _translate("MainWindow", "Time Frequency"))
        self.calibutton.setText(_translate("MainWindow", "Calibrate"))
        self.initbttn.setText(_translate("MainWindow", "Init Devs"))
        self.linkSensor.setText(_translate("MainWindow", "Link"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_calib), _translate("MainWindow", "Calibration"))
