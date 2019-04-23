# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1645, 893)
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(460, 0, 201, 61))
        self.label.setStyleSheet("font: 22pt \"Agency FB\";")
        self.label.setTextFormat(QtCore.Qt.PlainText)
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(1250, 810, 191, 31))
        self.pushButton.setObjectName("pushButton")
        self.groupBox = QtWidgets.QGroupBox(Form)
        self.groupBox.setGeometry(QtCore.QRect(1160, 50, 331, 131))
        self.groupBox.setObjectName("groupBox")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(20, 30, 54, 12))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setGeometry(QtCore.QRect(20, 80, 54, 12))
        self.label_3.setObjectName("label_3")
        self.colorLabel = QtWidgets.QLabel(self.groupBox)
        self.colorLabel.setGeometry(QtCore.QRect(80, 30, 71, 21))
        self.colorLabel.setObjectName("colorLabel")
        self.NumLabel = QtWidgets.QLabel(self.groupBox)
        self.NumLabel.setGeometry(QtCore.QRect(80, 70, 191, 41))
        self.NumLabel.setObjectName("NumLabel")
        self.groupBox_2 = QtWidgets.QGroupBox(Form)
        self.groupBox_2.setGeometry(QtCore.QRect(19, 49, 1121, 831))
        self.groupBox_2.setObjectName("groupBox_2")
        self.imgLabel = QtWidgets.QLabel(self.groupBox_2)
        self.imgLabel.setGeometry(QtCore.QRect(30, 30, 1071, 791))
        self.imgLabel.setObjectName("imgLabel")

        self.retranslateUi(Form)
        self.pushButton.clicked.connect(Form.openimage)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "图片显示"))
        self.pushButton.setText(_translate("Form", "选择图片"))
        self.groupBox.setTitle(_translate("Form", "车牌识别结果："))
        self.label_2.setText(_translate("Form", "颜色："))
        self.label_3.setText(_translate("Form", "车牌："))
        self.colorLabel.setText(_translate("Form", "绿色"))
        self.NumLabel.setText(_translate("Form", "浙A：12313123"))
        self.groupBox_2.setTitle(_translate("Form", "图片区域"))
        self.imgLabel.setText(_translate("Form", "TextLabel"))

