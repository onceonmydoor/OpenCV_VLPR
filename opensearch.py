# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'open.py'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtWidgets,QtGui,QtCore
from PyQt5.QtGui import QStandardItemModel,QStandardItem
import sys
from search import Ui_MainWindow #导入生成的界面类
from PyQt5.QtWidgets import QFileDialog,QApplication, QMainWindow, QWidget, QPushButton,QHeaderView,QMessageBox
from predict import Predict
import cv2
import openCamera
import qimage2ndarray
import pymysql



class Searchwindow(QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(Searchwindow,self).__init__()
        self.setupUi(self)
        self.conn = pymysql.connect(host='localhost',port=3306,user='root',password='root',db='opencv',charset='utf8')
        self.cur = self.conn.cursor()
        self.cur.execute("SELECT * FROM plate")
        self.showresult()

    def showresult(self):

        self.data = self.cur.fetchall()


        # 数据的列名
        self.col_lst = [tup[0] for tup in self.cur.description]
        row = len(self.data)
        if row==0:
            QMessageBox.critical(self, "抱歉", "没有能搜索到内容",
                                QMessageBox.Yes)
            return
        vol = len(self.data[0])
        self.tableView.horizontalHeader().setFont(QtGui.QFont('微软雅黑', 10))
        model = QStandardItemModel()
        title = ['ID', '车牌号码', '车牌颜色', '时间']
        model.setHorizontalHeaderLabels(title)
        model.setColumnCount(vol)
        model.setRowCount(row)
        for row, linedata in enumerate(self.data):
            for col, itemdata in enumerate(linedata):
                if itemdata == None:
                    item = QStandardItem("")
                else:
                    item = QStandardItem(str(itemdata))
                model.setItem(row, col, item)
        self.tableView.setModel(model)
        self.tableView.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)


    def GetsearchContent(self):
        #获取lineEdit控件的文本
        #self.lineEdit_2 = QtWidgets.QDateEdit()
        plate_num = self.lineEdit_2.text()
        print("车牌号码是："+plate_num)
        #获取dateEdit时间
        pre_datetime = self.dateTimeEdit.dateTime().toString("yyyy-MM-dd hh:mm:ss")
        last_datetime = self.dateTimeEdit_2.dateTime().toString("yyyy-MM-dd hh:mm:ss")
        #print(type(last_datetime))
        print("时间是从"+pre_datetime+"至"+last_datetime)
        #获取comboBox的内容
        #self.comboBox =QtWidgets.QComboBox()

        comboBox_contents = self.comboBox.currentText()
        print("颜色是"+comboBox_contents)
        if pre_datetime > last_datetime:
            QMessageBox.warning(self, "警告", "初始时间大于截至时间，请重新输入！",
                                QMessageBox.Yes)
        else:
            self.cur.execute(' SELECT * FROM plate WHERE time BETWEEN "' + pre_datetime + '" AND "' + last_datetime + '" ')
            self.showresult()

        if plate_num:
            self.cur.execute(' SELECT * FROM plate WHERE plate_num = "'+plate_num+'" ')
            self.showresult()

        if comboBox_contents!="全部":
            self.cur.execute(' SELECT * FROM plate WHERE plate_color = "' + comboBox_contents + '" ')
            self.showresult()







if __name__ == '__main__':

    app =QtWidgets.QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    window = Searchwindow()
    window.show()
    sys.exit(app.exec_())