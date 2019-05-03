# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'search.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(783, 591)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tableView = QtWidgets.QTableView(self.centralwidget)
        self.tableView.setGeometry(QtCore.QRect(10, 60, 761, 491))
        self.tableView.setObjectName("tableView")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(40, 10, 121, 31))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.dateTimeEdit = QtWidgets.QDateTimeEdit(self.centralwidget)
        self.dateTimeEdit.setGeometry(QtCore.QRect(340, 10, 141, 31))
        self.dateTimeEdit.setDateTime(QtCore.QDateTime(QtCore.QDate(2019, 5, 1), QtCore.QTime(0, 0, 0)))
        self.dateTimeEdit.setObjectName("dateTimeEdit")
        self.dateTimeEdit_2 = QtWidgets.QDateTimeEdit(self.centralwidget)
        self.dateTimeEdit_2.setGeometry(QtCore.QRect(510, 10, 141, 31))
        self.dateTimeEdit_2.setDateTime(QtCore.QDateTime(QtCore.QDate(2019, 5, 1), QtCore.QTime(0, 0, 0)))
        self.dateTimeEdit_2.setObjectName("dateTimeEdit_2")
        self.search = QtWidgets.QPushButton(self.centralwidget)
        self.search.setGeometry(QtCore.QRect(660, 10, 91, 31))
        self.search.setObjectName("search")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(3, 11, 41, 31))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(490, 10, 21, 31))
        self.label_2.setObjectName("label_2")
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(210, 10, 71, 31))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(300, 20, 31, 16))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(170, 10, 41, 31))
        self.label_4.setObjectName("label_4")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 783, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.search.clicked.connect(MainWindow.GetsearchContent)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.dateTimeEdit.setDisplayFormat(_translate("MainWindow", "yyyy/MM/dd HH:mm:ss"))
        self.dateTimeEdit_2.setDisplayFormat(_translate("MainWindow", "yyyy/MM/dd HH:mm:ss"))
        self.search.setText(_translate("MainWindow", " 查询"))
        self.label.setText(_translate("MainWindow", "车牌："))
        self.label_2.setText(_translate("MainWindow", "至"))
        self.comboBox.setItemText(0, _translate("MainWindow", "全部"))
        self.comboBox.setItemText(1, _translate("MainWindow", "绿色"))
        self.comboBox.setItemText(2, _translate("MainWindow", "黄色"))
        self.comboBox.setItemText(3, _translate("MainWindow", "蓝色"))
        self.label_3.setText(_translate("MainWindow", "时间："))
        self.label_4.setText(_translate("MainWindow", "颜色"))

