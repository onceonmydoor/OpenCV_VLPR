# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'open.py'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtWidgets,QtGui,QtCore
import sys
from test import Ui_Form #导入生成的界面类
from PyQt5.QtWidgets import QFileDialog,QApplication, QMainWindow, QWidget, QPushButton
from predict import Predict
import cv2
import openCamera



class mywindow(QtWidgets.QWidget,Ui_Form):
    def __init__(self):
        super(mywindow,self).__init__()
        self.setupUi(self)
        #定义槽函数
    def openimage(self):
        Eng2Chi = {"green":"绿色","blue":"蓝色","yellow":"黄色"}
    #打开文件路径
    #设置文件扩展名过滤，注意用双分号间隔
        imgPath , imgType = QFileDialog.getOpenFileName(self,"打开图片","","*.jpg;;*.png;;*.jpeg;;*.bmp;;All Files (*)")
        print(imgPath)
        #利用qlabel显示图片
        png = QtGui.QPixmap(imgPath).scaled(self.imgLabel.width(),self.imgLabel.height())
        self.imgLabel.setPixmap(png)


        ###################识别#####################
        q = Predict()
        afterprocess, old = q.preprocess(imgPath)
        colors, card_imgs = q.locate_carPlate(afterprocess, old)
        result, roi, color, divs = q.char_recogize(colors, card_imgs)  # all list
        if len(result) == 0:
            print("未能识别到车牌")
            self.colorLabel.setText("抱歉未能识别到车牌")
            self.NumLabel.setText("抱歉，未能识别到车牌")
        else:
            for r  in range(len(result)):
                print("#"*10+"识别结果是"+"#"*10)
                print("车牌的颜色为：" + Eng2Chi[color[r]])
                self.colorLabel.setText(Eng2Chi[color[r]])
                print(result[r])
                result[r].insert(2,"-")
                self.NumLabel.setText(''.join(result[r]))
                print("#" * 25)
                roi[r] = cv2.cvtColor(roi[r],cv2.COLOR_BGR2RGB)
                QtImg = QtGui.QImage(roi[r].data,roi[r].shape[1],roi[r].shape[0],QtGui.QImage.Format_RGB888)
                #显示图片到label中
                self.location.resize(QtCore.QSize(roi[r].shape[1],roi[r].shape[0]))
                self.location.setPixmap(QtGui.QPixmap.fromImage(QtImg))

                print("\n")
        ###################识别#####################


        if len(divs)==0:
            pass
        elif len(divs[0])<7:
            pass
        else:
            # 与上同理
            Gray1 = self.QImage2Pixmap(0,divs)
            #self.div1.resize(QtCore.QSize(divs[0][0].shape[1], divs[0][0].shape[0]))
            self.div1.setPixmap(QtGui.QPixmap.fromImage(Gray1))

            Gray2 = self.QImage2Pixmap(1, divs)
            # self.div1.resize(QtCore.QSize(divs[0][0].shape[1], divs[0][0].shape[0]))
            self.div2.setPixmap(QtGui.QPixmap.fromImage(Gray2))

            Gray3 = self.QImage2Pixmap(2, divs)
            # self.div1.resize(QtCore.QSize(divs[0][0].shape[1], divs[0][0].shape[0]))
            self.div3.setPixmap(QtGui.QPixmap.fromImage(Gray3))

            Gray4 = self.QImage2Pixmap(3, divs)
            # self.div1.resize(QtCore.QSize(divs[0][0].shape[1], divs[0][0].shape[0]))
            self.div4.setPixmap(QtGui.QPixmap.fromImage(Gray4))

            Gray5 = self.QImage2Pixmap(4, divs)
            # self.div1.resize(QtCore.QSize(divs[0][0].shape[1], divs[0][0].shape[0]))
            self.div5.setPixmap(QtGui.QPixmap.fromImage(Gray5))

            Gray6 = self.QImage2Pixmap(5, divs)
            # self.div1.resize(QtCore.QSize(divs[0][0].shape[1], divs[0][0].shape[0]))
            self.div6.setPixmap(QtGui.QPixmap.fromImage(Gray6))

            Gray7 = self.QImage2Pixmap(6, divs)
            # self.div1.resize(QtCore.QSize(divs[0][0].shape[1], divs[0][0].shape[0]))
            self.div7.setPixmap(QtGui.QPixmap.fromImage(Gray7))





    def QImage2Pixmap(self,i,divs):
        return QtGui.QImage(divs[0][i].data, divs[0][i].shape[1], divs[0][i].shape[0],
                             QtGui.QImage.Format_Grayscale8)

    def slot_btn_function(self):
        self.hide()
        self.s = openCamera.CamShow()
        self.s.show()


# class camera_Form(QWidget):
#     def __init(self):
#         super(camera_Form,self).__init__()
#         self.init_ui()
#     def init_ui(self):
#         self.resize(2000,1000)
#         self.setWindowTitle("camera")
#         self.btn = QPushButton("jump",self)
#         self.btn.setGeometry(150, 150, 100, 50)
#         self.btn.clicked.connect(self.slot_btn_function1)
#
#     def slot_btn_function1(self):
#         self.hide()
#         self.f = Ui_Form()
#         self.f.show()

if __name__ == '__main__':

    app =QtWidgets.QApplication(sys.argv)
    window = mywindow()
    window.show()
    sys.exit(app.exec_())