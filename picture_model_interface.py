# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'picture_model_interface.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1549, 910)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("C:/Users/dell.DESKTOP-FM0VES7/.designer/backup/icon/图片.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Form.setWindowIcon(icon)
        self.frame = QtWidgets.QFrame(Form)
        self.frame.setGeometry(QtCore.QRect(30, 10, 1111, 921))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.groupBox_2 = QtWidgets.QGroupBox(self.frame)
        self.groupBox_2.setGeometry(QtCore.QRect(0, 40, 1111, 851))
        self.groupBox_2.setObjectName("groupBox_2")
        self.imgLabel = QtWidgets.QLabel(self.groupBox_2)
        self.imgLabel.setGeometry(QtCore.QRect(20, 20, 1081, 821))
        self.imgLabel.setText("")
        self.imgLabel.setObjectName("imgLabel")
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setGeometry(QtCore.QRect(470, 0, 131, 51))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(23)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setStyleSheet("font: 23pt \"微软雅黑\";")
        self.label.setTextFormat(QtCore.Qt.PlainText)
        self.label.setObjectName("label")
        self.frame_2 = QtWidgets.QFrame(Form)
        self.frame_2.setGeometry(QtCore.QRect(1140, 10, 381, 901))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.groupBox = QtWidgets.QGroupBox(self.frame_2)
        self.groupBox.setGeometry(QtCore.QRect(30, 30, 341, 161))
        self.groupBox.setStyleSheet("")
        self.groupBox.setObjectName("groupBox")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(20, 30, 54, 12))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setGeometry(QtCore.QRect(20, 80, 54, 12))
        self.label_3.setObjectName("label_3")
        self.colorLabel = QtWidgets.QLabel(self.groupBox)
        self.colorLabel.setGeometry(QtCore.QRect(80, 20, 91, 31))
        self.colorLabel.setStyleSheet("font: 20pt \"Agency FB\";\n"
"font: rgb(85, 255, 127)")
        self.colorLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.colorLabel.setObjectName("colorLabel")
        self.NumLabel = QtWidgets.QLabel(self.groupBox)
        self.NumLabel.setGeometry(QtCore.QRect(80, 60, 241, 51))
        self.NumLabel.setStyleSheet("font: 30pt \"Open Sans\";")
        self.NumLabel.setObjectName("NumLabel")
        self.groupBox_3 = QtWidgets.QGroupBox(self.frame_2)
        self.groupBox_3.setGeometry(QtCore.QRect(30, 210, 341, 171))
        self.groupBox_3.setObjectName("groupBox_3")
        self.location = QtWidgets.QLabel(self.groupBox_3)
        self.location.setGeometry(QtCore.QRect(20, 40, 281, 91))
        self.location.setObjectName("location")
        self.groupBox_4 = QtWidgets.QGroupBox(self.frame_2)
        self.groupBox_4.setGeometry(QtCore.QRect(30, 400, 341, 181))
        self.groupBox_4.setObjectName("groupBox_4")
        self.div1 = QtWidgets.QLabel(self.groupBox_4)
        self.div1.setGeometry(QtCore.QRect(10, 70, 31, 51))
        self.div1.setText("")
        self.div1.setObjectName("div1")
        self.div2 = QtWidgets.QLabel(self.groupBox_4)
        self.div2.setGeometry(QtCore.QRect(50, 70, 31, 51))
        self.div2.setText("")
        self.div2.setObjectName("div2")
        self.div3 = QtWidgets.QLabel(self.groupBox_4)
        self.div3.setGeometry(QtCore.QRect(100, 70, 31, 51))
        self.div3.setText("")
        self.div3.setObjectName("div3")
        self.div4 = QtWidgets.QLabel(self.groupBox_4)
        self.div4.setGeometry(QtCore.QRect(140, 70, 31, 51))
        self.div4.setText("")
        self.div4.setObjectName("div4")
        self.div5 = QtWidgets.QLabel(self.groupBox_4)
        self.div5.setGeometry(QtCore.QRect(190, 70, 31, 51))
        self.div5.setText("")
        self.div5.setObjectName("div5")
        self.div6 = QtWidgets.QLabel(self.groupBox_4)
        self.div6.setGeometry(QtCore.QRect(230, 70, 31, 51))
        self.div6.setText("")
        self.div6.setObjectName("div6")
        self.div7 = QtWidgets.QLabel(self.groupBox_4)
        self.div7.setGeometry(QtCore.QRect(270, 70, 31, 51))
        self.div7.setText("")
        self.div7.setObjectName("div7")
        self.div8 = QtWidgets.QLabel(self.groupBox_4)
        self.div8.setGeometry(QtCore.QRect(310, 70, 31, 51))
        self.div8.setText("")
        self.div8.setObjectName("div8")
        self.pushButton = QtWidgets.QPushButton(self.frame_2)
        self.pushButton.setGeometry(QtCore.QRect(30, 610, 191, 31))
        self.pushButton.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.pushButton.setStyleSheet("background-color:rgb(94, 129, 172);\n"
"color:white;\n"
"font: 10pt \"微软雅黑\";\n"
"border-radius:7px;")
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.frame_2)
        self.pushButton_2.setGeometry(QtCore.QRect(30, 680, 191, 31))
        self.pushButton_2.setStyleSheet("background-color:rgb(94, 129, 172);\n"
"color:white;\n"
"font: 10pt \"微软雅黑\";\n"
"border-radius:7px;")
        self.pushButton_2.setObjectName("pushButton_2")
        self.sql_model = QtWidgets.QPushButton(self.frame_2)
        self.sql_model.setGeometry(QtCore.QRect(30, 750, 191, 31))
        self.sql_model.setStyleSheet("background-color:rgb(94, 129, 172);\n"
"color:white;\n"
"font: 10pt \"微软雅黑\";\n"
"border-radius:7px;")
        self.sql_model.setObjectName("sql_model")
        self.pushButton_3 = QtWidgets.QPushButton(self.frame_2)
        self.pushButton_3.setGeometry(QtCore.QRect(30, 820, 191, 31))
        self.pushButton_3.setStyleSheet("background-color:rgb(94, 129, 172);\n"
"color:white;\n"
"font: 10pt \"微软雅黑\";\n"
"border-radius:7px;")
        self.pushButton_3.setObjectName("pushButton_3")

        self.retranslateUi(Form)
        self.pushButton.clicked.connect(Form.openimage)
        self.pushButton_2.clicked.connect(Form.slot_btn_function)
        self.sql_model.clicked.connect(Form.sql_btn_function)
        self.pushButton_3.clicked.connect(Form.show_preprocess)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "单图模式"))
        self.groupBox_2.setTitle(_translate("Form", "图片区域"))
        self.label.setText(_translate("Form", "图片显示"))
        self.groupBox.setTitle(_translate("Form", "车牌识别结果："))
        self.label_2.setText(_translate("Form", "颜色："))
        self.label_3.setText(_translate("Form", "车牌："))
        self.colorLabel.setText(_translate("Form", "绿色"))
        self.NumLabel.setText(_translate("Form", "浙A-XXXXXXX"))
        self.groupBox_3.setTitle(_translate("Form", "定位到的车牌"))
        self.location.setText(_translate("Form", "车牌定位图"))
        self.groupBox_4.setTitle(_translate("Form", "分割结果"))
        self.pushButton.setText(_translate("Form", "选择图片"))
        self.pushButton_2.setText(_translate("Form", "开启摄像头模式"))
        self.sql_model.setText(_translate("Form", "数据库查询"))
        self.pushButton_3.setText(_translate("Form", "查看预处理图片"))

