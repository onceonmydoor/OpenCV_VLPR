# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication,QMainWindow,QWidget,QPushButton

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1675, 1012)
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(460, 0, 201, 61))
        self.label.setStyleSheet("font: 22pt \"Agency FB\";")
        self.label.setTextFormat(QtCore.Qt.PlainText)
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(1170, 630, 191, 31))
        self.pushButton.setObjectName("pushButton")
        self.groupBox = QtWidgets.QGroupBox(Form)
        self.groupBox.setGeometry(QtCore.QRect(1160, 50, 501, 161))
        self.groupBox.setObjectName("groupBox")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(20, 30, 54, 12))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setGeometry(QtCore.QRect(20, 80, 54, 12))
        self.label_3.setObjectName("label_3")
        self.colorLabel = QtWidgets.QLabel(self.groupBox)
        self.colorLabel.setGeometry(QtCore.QRect(80, 20, 71, 31))
        self.colorLabel.setStyleSheet("font: 75 10pt \"Agency FB\";\n"
"font: 9pt \"Algerian\";\n"
"font: 11pt \"Agency FB\";\n"
"font: rgb(85, 255, 127)")
        self.colorLabel.setObjectName("colorLabel")
        self.NumLabel = QtWidgets.QLabel(self.groupBox)
        self.NumLabel.setGeometry(QtCore.QRect(80, 60, 231, 51))
        self.NumLabel.setStyleSheet("font: 14pt \"Agency FB\";\n"
"font: 16pt \"Agency FB\";\n"
"font: 20pt \"Agency FB\";\n"
"font: 36pt \"Agency FB\";")
        self.NumLabel.setObjectName("NumLabel")
        self.groupBox_2 = QtWidgets.QGroupBox(Form)
        self.groupBox_2.setGeometry(QtCore.QRect(19, 49, 1121, 941))
        self.groupBox_2.setObjectName("groupBox_2")
        self.imgLabel = QtWidgets.QLabel(self.groupBox_2)
        self.imgLabel.setGeometry(QtCore.QRect(30, 20, 1071, 901))
        self.imgLabel.setObjectName("imgLabel")
        self.groupBox_3 = QtWidgets.QGroupBox(Form)
        self.groupBox_3.setGeometry(QtCore.QRect(1160, 220, 501, 171))
        self.groupBox_3.setObjectName("groupBox_3")
        self.location = QtWidgets.QLabel(self.groupBox_3)
        self.location.setGeometry(QtCore.QRect(20, 30, 221, 61))
        self.location.setObjectName("location")
        self.groupBox_4 = QtWidgets.QGroupBox(Form)
        self.groupBox_4.setGeometry(QtCore.QRect(1160, 430, 501, 161))
        self.groupBox_4.setObjectName("groupBox_4")
        self.div1 = QtWidgets.QLabel(self.groupBox_4)
        self.div1.setGeometry(QtCore.QRect(10, 50, 31, 41))
        self.div1.setObjectName("div1")
        self.div2 = QtWidgets.QLabel(self.groupBox_4)
        self.div2.setGeometry(QtCore.QRect(50, 50, 31, 41))
        self.div2.setObjectName("div2")
        self.div3 = QtWidgets.QLabel(self.groupBox_4)
        self.div3.setGeometry(QtCore.QRect(100, 50, 31, 41))
        self.div3.setObjectName("div3")
        self.div4 = QtWidgets.QLabel(self.groupBox_4)
        self.div4.setGeometry(QtCore.QRect(140, 50, 31, 41))
        self.div4.setObjectName("div4")
        self.div5 = QtWidgets.QLabel(self.groupBox_4)
        self.div5.setGeometry(QtCore.QRect(180, 50, 31, 41))
        self.div5.setObjectName("div5")
        self.div6 = QtWidgets.QLabel(self.groupBox_4)
        self.div6.setGeometry(QtCore.QRect(220, 50, 31, 41))
        self.div6.setObjectName("div6")
        self.div7 = QtWidgets.QLabel(self.groupBox_4)
        self.div7.setGeometry(QtCore.QRect(260, 50, 31, 41))
        self.div7.setObjectName("div7")
        self.div8 = QtWidgets.QLabel(self.groupBox_4)
        self.div8.setGeometry(QtCore.QRect(300, 50, 31, 41))
        self.div8.setObjectName("div8")
        self.pushButton_2 = QtWidgets.QPushButton(Form)
        self.pushButton_2.setGeometry(QtCore.QRect(1170, 690, 191, 31))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(Form)
        self.pushButton_3.setGeometry(QtCore.QRect(1170, 750, 191, 31))
        self.pushButton_3.setObjectName("pushButton_3")

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
        self.NumLabel.setText(_translate("Form", "浙A-XXXXXXX"))
        self.groupBox_2.setTitle(_translate("Form", "图片区域"))
        self.imgLabel.setText(_translate("Form", "TextLabel"))
        self.groupBox_3.setTitle(_translate("Form", "定位到的车牌"))
        self.location.setText(_translate("Form", "车牌定位图"))
        self.groupBox_4.setTitle(_translate("Form", "分割结果"))
        self.div1.setText(_translate("Form", "1"))
        self.div2.setText(_translate("Form", "2"))
        self.div3.setText(_translate("Form", "3"))
        self.div4.setText(_translate("Form", "4"))
        self.div5.setText(_translate("Form", "5"))
        self.div6.setText(_translate("Form", "6"))
        self.div7.setText(_translate("Form", "7"))
        self.div8.setText(_translate("Form", "8"))
        self.pushButton_2.setText(_translate("Form", "开启摄像头模式"))
        self.pushButton_3.setText(_translate("Form", "查看预处理图片"))



