# -*- coding: utf-8 -*-
from Ui_main2 import Ui_MainWindow
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QDesktopWidget,
    QWidget,
    QStackedLayout,
    QSlider,
    QDoubleSpinBox,
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QSpinBox,
    QAbstractItemView,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
import sys
import yaml
from app_function import callBack
import os
from Ui_main2 import Ui_MainWindow


class AudioWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.items = self.initFile()
        self.function = callBack(self)
        self.setupUi(self)
        self.initUI()
        self.initWidget()

    def initUI(self):
        with open("./style/ui.qss", mode="r", encoding="utf-8") as f:
            self.setStyleSheet(f.read())
        center_point = QDesktopWidget().availableGeometry().center()
        x = center_point.x()
        y = center_point.y()
        _, _, width, height = self.frameGeometry().getRect()
        self.move(x - width // 2, y - height // 2)
        self.setFixedSize(width, height)
        self.setWindowIcon(QIcon("./images/千绘莉.ico"))

    def initWidget(self):
        self.comboBox.addItems(self.items["Engine"])
        self.comboBox.setCurrentIndex(1)
        self.comboBox_2.addItems([i[0] for i in self.items["Bert-Vits"].items()])
        self.comboBox_2.setCurrentIndex(2)
        self.comboBox_3.addItems(
            [i[0] for i in self.items["Language"]["Bert-Vits"].items()]
        )
        self.comboBox_3.setCurrentIndex(0)
        self.comboBox.currentIndexChanged.connect(self.function.onComboBoxIndexChanged)
        self.comboBox_3.currentIndexChanged.connect(
            self.function.onComboBox_3IndexChanged
        )
        self.pushButton_2.clicked.connect(self.function.onPushButton_2Clicked)
        self.pushButton_3.clicked.connect(self.function.clearBrowser_2)
        self.pushButton_4.clicked.connect(self.function.clearBrowser)
        self.pushButton.clicked.connect(lambda: self.function.onPushButtonClicked(1))
        self.pushButton_5.clicked.connect(self.function.onPushButton_5Clicked)
        self.pushButton_6.clicked.connect(self.function.onPushButton_6Clicked)
        self.tabWidget.setCurrentIndex(0)
        self.stackedLayout = QStackedLayout()
        self.stackedLayout.setObjectName("stackedLayout")
        self.widget = QWidget()
        self.stackedLayout.addWidget(self.widget)
        self.widget.setObjectName("widget")
        self.gridLayout_2 = QGridLayout()
        self.label_4 = QLabel("音量")
        self.label_4.setAlignment(Qt.AlignCenter)
        self.gridLayout_2.addWidget(self.label_4, 0, 0, 1, 1)
        self.label_5 = QLabel("音调")
        self.label_5.setAlignment(Qt.AlignCenter)
        self.gridLayout_2.addWidget(self.label_5, 1, 0, 1, 1)
        self.label_6 = QLabel("语速")
        self.label_6.setAlignment(Qt.AlignCenter)
        self.gridLayout_2.addWidget(self.label_6, 2, 0, 1, 1)
        self.HSlider_2 = QSlider()
        self.HSlider_2.setOrientation(Qt.Horizontal)
        self.HSlider_2.setMinimum(0)
        self.HSlider_2.setMaximum(15)
        self.HSlider_2.setValue(5)
        self.gridLayout_2.addWidget(self.HSlider_2, 0, 1, 1, 1)
        self.HSlider_3 = QSlider()
        self.HSlider_3.setOrientation(Qt.Horizontal)
        self.HSlider_3.setMinimum(0)
        self.HSlider_3.setMaximum(15)
        self.HSlider_3.setValue(5)
        self.gridLayout_2.addWidget(self.HSlider_3, 1, 1, 1, 1)
        self.HSlider_4 = QSlider()
        self.HSlider_4.setOrientation(Qt.Horizontal)
        self.HSlider_4.setMinimum(0)
        self.HSlider_4.setMaximum(15)
        self.HSlider_4.setValue(5)
        self.gridLayout_2.addWidget(self.HSlider_4, 2, 1, 1, 1)
        self.gridLayout_2.setSpacing(10)
        self.gridLayout_2.setColumnStretch(0, 1)
        self.gridLayout_2.setColumnStretch(1, 2)
        self.widget.setLayout(self.gridLayout_2)
        self.widget_2 = QWidget()
        self.widget_2.setObjectName("widget_2")
        self.horizontalLayout_8 = QHBoxLayout()
        self.widget_2.setLayout(self.horizontalLayout_8)
        self.groupBox_4 = QGroupBox("音量")
        self.horizontalLayout_8.addWidget(self.groupBox_4)
        self.horizontalLayout_9 = QHBoxLayout()
        self.groupBox_4.setLayout(self.horizontalLayout_9)
        self.HSlider = QSlider()
        self.horizontalLayout_9.addWidget(self.HSlider)
        self.HSlider.setOrientation(Qt.Horizontal)
        self.HSlider.setMinimum(0)
        self.HSlider.setMaximum(100)
        self.HSlider.setValue(50)
        self.groupBox_5 = QGroupBox("语速")
        self.horizontalLayout_8.addWidget(self.groupBox_5)
        self.horizontalLayout_10 = QHBoxLayout()
        self.groupBox_5.setLayout(self.horizontalLayout_10)
        self.doubleSpinBox = QDoubleSpinBox()
        self.doubleSpinBox.setRange(0.5, 2.0)
        self.doubleSpinBox.setPrefix("x")
        self.doubleSpinBox.setDecimals(1)
        self.doubleSpinBox.setSingleStep(0.1)
        self.doubleSpinBox.setValue(1.0)
        self.horizontalLayout_10.addWidget(self.doubleSpinBox)
        self.stackedLayout.addWidget(self.widget_2)
        self.widget_3 = QWidget()
        self.widget_3.setObjectName("widget_3")
        self.horizontalLayout_12 = QHBoxLayout()
        self.widget_3.setLayout(self.horizontalLayout_12)
        self.groupBox_9 = QGroupBox("音量")
        self.horizontalLayout_12.addWidget(self.groupBox_9)
        self.horizontalLayout_13 = QHBoxLayout()
        self.groupBox_9.setLayout(self.horizontalLayout_13)
        self.HSlider_5 = QSlider()
        self.horizontalLayout_13.addWidget(self.HSlider_5)
        self.HSlider_5.setOrientation(Qt.Horizontal)
        self.HSlider_5.setMinimum(0)
        self.HSlider_5.setMaximum(100)
        self.HSlider_5.setValue(50)
        self.groupBox_10 = QGroupBox("语速")
        self.horizontalLayout_12.addWidget(self.groupBox_10)
        self.horizontalLayout_14 = QHBoxLayout()
        self.groupBox_10.setLayout(self.horizontalLayout_14)
        self.doubleSpinBox_2 = QDoubleSpinBox()
        self.doubleSpinBox_2.setRange(0.5, 2.0)
        self.doubleSpinBox_2.setPrefix("x")
        self.doubleSpinBox_2.setDecimals(1)
        self.doubleSpinBox_2.setSingleStep(0.1)
        self.doubleSpinBox_2.setValue(1.0)
        self.horizontalLayout_14.addWidget(self.doubleSpinBox_2)
        self.stackedLayout.addWidget(self.widget_3)
        self.widget_4 = QWidget()
        self.widget_4.setObjectName("widget_4")
        self.horizontalLayout_11 = QHBoxLayout()
        self.groupBox_6 = QGroupBox("语速")
        self.horizontalLayout_11.addWidget(self.groupBox_6)
        self.spinBox = QSpinBox()
        self.spinBox.setRange(-100, 200)
        self.spinBox.setSuffix("%")
        self.spinBox.setSingleStep(10)
        self.spinBox.setValue(0)
        self.verticalLayout_6 = QVBoxLayout()
        self.verticalLayout_6.addWidget(self.spinBox)
        self.groupBox_6.setLayout(self.verticalLayout_6)
        self.groupBox_7 = QGroupBox("音量")
        self.horizontalLayout_11.addWidget(self.groupBox_7)
        self.spinBox_2 = QSpinBox()
        self.spinBox_2.setRange(-100, 100)
        self.spinBox_2.setSuffix("%")
        self.spinBox_2.setSingleStep(1)
        self.spinBox_2.setValue(0)
        self.verticalLayout_7 = QVBoxLayout()
        self.verticalLayout_7.addWidget(self.spinBox_2)
        self.groupBox_7.setLayout(self.verticalLayout_7)
        self.groupBox_8 = QGroupBox("音调")
        self.horizontalLayout_11.addWidget(self.groupBox_8)
        self.spinBox_3 = QSpinBox()
        self.spinBox_3.setRange(-200, 200)
        self.spinBox_3.setSuffix("Hz")
        self.spinBox_3.setSingleStep(10)
        self.spinBox_3.setValue(0)
        self.verticalLayout_8 = QVBoxLayout()
        self.verticalLayout_8.addWidget(self.spinBox_3)
        self.groupBox_8.setLayout(self.verticalLayout_8)
        self.horizontalLayout_11.setSpacing(0)
        self.widget_4.setLayout(self.horizontalLayout_11)
        self.stackedLayout.addWidget(self.widget_4)
        self.stackedLayout.setCurrentIndex(1)
        self.groupBox_2.setLayout(self.stackedLayout)
        self.pushButton_8.clicked.connect(self.function.onPushButton_8Clicked)
        self.lineEdit.setText("./audio/audio.wav")
        self.lineEdit.setReadOnly(True)
        self.pushButton_5.setEnabled(False)
        self.pushButton_6.setEnabled(False)
        self.listWidget.addItems(["./audio/" + i for i in os.listdir("./audio")])
        self.listWidget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.pushButton_7.clicked.connect(self.function.onPushButton_7Clicked)
        self.pushButton_5.setIcon(QIcon("./images/cil-media-pause.png"))
        self.pushButton_6.setIcon(QIcon("./images/cil-media-play.png"))
        self.pushButton_7.setIcon(QIcon("./images/cil-media-skip-forward.png"))
        self.pushButton_9.clicked.connect(self.function.onPushButton_9Clicked)
        self.pushButton_11.clicked.connect(self.function.onPushButton_11Clicked)
        self.pushButton_10.clicked.connect(self.function.onpushButton_10Clicked)

    def initFile(self):
        with open("./config/items.yaml", "r", encoding="utf-8") as f:
            items = yaml.load(f.read(), Loader=yaml.FullLoader)
        return items


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioWindow()
    window.show()
    sys.exit(app.exec_())
