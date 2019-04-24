# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'open.py'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtWidgets,QtGui,QtCore
import sys
from test import Ui_Form #导入生成的界面类
from PyQt5.QtWidgets import QFileDialog
from predict import Predict
import cv2



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
                # 同理

        QtImg = QtGui.QImage(divs[0][0].data, divs[0][0].shape[1], divs[0][0].shape[0],
                             QtGui.QImage.Format_RGB888)
        # 显示图片到label中
        self.div1.resize(QtCore.QSize(divs[0][0].shape[1], divs[0][0].shape[0]))
        self.div1.setPixmap(QtGui.QPixmap.fromImage(QtImg))

if __name__ == '__main__':

    app =QtWidgets.QApplication(sys.argv)
    window = mywindow()
    window.show()
    sys.exit(app.exec_())