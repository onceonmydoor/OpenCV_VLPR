# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'open.py'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtWidgets,QtGui,QtCore
import sys
from picture_model_interface import Ui_Form #导入生成的界面类
from PyQt5.QtWidgets import QFileDialog,QApplication, QMainWindow, QWidget, QPushButton
from predict import Predict
import cv2
import openCamera
import openSearch
import qimage2ndarray
import pymysql
import datetime
import qdarkstyle




class mywindow(QtWidgets.QWidget,Ui_Form):
    def __init__(self):
        super(mywindow,self).__init__()
        self.setupUi(self)
        #定义槽函数
        self.Img_preprocess = None


    def openimage(self):
        Eng2Chi = {"green":"绿色","blue":"蓝色","yellow":"黄色"}
        #打开文件路径
        #设置文件扩展名过滤，注意用双分号间隔
        imgPath , imgType = QFileDialog.getOpenFileName(self,"打开图片","","*.jpg;;*.png;;*.jpeg;;*.bmp;;All Files (*)")
        print(imgPath)
        #利用qlabel显示图片
        png = QtGui.QPixmap(imgPath).scaled(self.imgLabel.width(),self.imgLabel.height())
        self.imgLabel.setPixmap(png)

        if imgPath:
            ###################识别#####################
            q = Predict()
            afterprocess, old = q.preprocess(imgPath)
            self.Img_preprocess = afterprocess
            colors, card_imgs = q.locate_carPlate(afterprocess, old)
            result, roi, color, divs = q.char_recogize(colors, card_imgs)  # all list
            if len(result) == 0:
                print("未能识别到车牌")
                self.colorLabel.setText("抱歉未能识别到车牌")
                self.NumLabel.setText("抱歉，未能识别到车牌")
                self.location.clear()
            else:
                for r  in range(len(result)):
                    final_result = ''.join(result[r])
                    if len(final_result)<7:
                        self.NumLabel.setText("抱歉，未能识别到车牌")
                        continue

                    print("#"*10+"识别结果是"+"#"*10)
                    print("车牌的颜色为：" + Eng2Chi[color[r]])
                    final_color = Eng2Chi[color[r]]
                    if final_color == "蓝色":
                        self.colorLabel.setStyleSheet("QLabel{background-color:blue;color:white}")
                        #self.colorLabel.setStyleSheet("color:white")
                    self.colorLabel.setText(final_color)
                    print(result[r])
                    result[r].insert(2,"-")
                    self.NumLabel.setText(final_result)
                    print("#" * 25)
                    roi[r] = cv2.cvtColor(roi[r],cv2.COLOR_BGR2RGB)
                    qimg = qimage2ndarray.array2qimage(roi[r])
                    local_img =qimg.scaled(self.location.width(),self.location.height())
                    self.location.setPixmap(QtGui.QPixmap(local_img))
                    # QtImg = QtGui.QImage(roi[r].data,roi[r].shape[1],roi[r].shape[0],QtGui.QImage.Format_RGB888)
                    # #显示图片到label中
                    # self.location.resize(QtCore.QSize(roi[r].shape[1],roi[r].shape[0]))
                    # self.location.setPixmap(QtGui.QPixmap.fromImage(QtImg))



                    if len(final_result)>=8 and u'\u4e00' <= final_result[0] <= u'\u9fff':
                        #保存至MYSQL数据库
                        conn, cur = self.connetSQL()
                        currenttime = datetime.datetime.now()
                        print(currenttime.strftime('%Y-%m-%d %H:%M:%S'))
                        try:
                            cur.execute("insert into plate(plate_num,plate_color,time)values(%s,%s,%s)",(final_result,final_color,currenttime))
                            conn.commit()
                        except Exception as e:
                            print("expect: ",e)
                        finally:
                            cur.close()
                            conn.close()






            ###################识别#####################


            if len(divs)==0:
                pass
            elif len(divs[0])<7:
                pass
            else:
                # 与上同理
                Gray1 = self.QImage2Pixmap(0,divs)
                #self.div1.resize(QtCore.QSize(divs[0][0].shape[1], divs[0][0].shape[0]))
                Gray1 = Gray1.scaled(self.div1.width(),self.div1.height())
                self.div1.setPixmap(QtGui.QPixmap.fromImage(Gray1))

                Gray2 = self.QImage2Pixmap(1, divs)
                # self.div1.resize(QtCore.QSize(divs[0][0].shape[1], divs[0][0].shape[0]))
                Gray2 = Gray2.scaled(self.div2.width(), self.div2.height())
                self.div2.setPixmap(QtGui.QPixmap.fromImage(Gray2))

                Gray3 = self.QImage2Pixmap(2, divs)
                # self.div1.resize(QtCore.QSize(divs[0][0].shape[1], divs[0][0].shape[0]))
                Gray3 = Gray3.scaled(self.div3.width(), self.div3.height())
                self.div3.setPixmap(QtGui.QPixmap.fromImage(Gray3))

                Gray4 = self.QImage2Pixmap(3, divs)
                # self.div1.resize(QtCore.QSize(divs[0][0].shape[1], divs[0][0].shape[0]))
                Gray4 = Gray4.scaled(self.div4.width(), self.div4.height())
                self.div4.setPixmap(QtGui.QPixmap.fromImage(Gray4))

                Gray5 = self.QImage2Pixmap(4, divs)
                # self.div1.resize(QtCore.QSize(divs[0][0].shape[1], divs[0][0].shape[0]))
                Gray5 = Gray5.scaled(self.div5.width(), self.div5.height())
                self.div5.setPixmap(QtGui.QPixmap.fromImage(Gray5))

                Gray6 = self.QImage2Pixmap(5, divs)
                # self.div1.resize(QtCore.QSize(divs[0][0].shape[1], divs[0][0].shape[0]))
                Gray6 = Gray6.scaled(self.div6.width(), self.div6.height())
                self.div6.setPixmap(QtGui.QPixmap.fromImage(Gray6))

                Gray7 = self.QImage2Pixmap(6, divs)
                # self.div1.resize(QtCore.QSize(divs[0][0].shape[1], divs[0][0].shape[0]))
                Gray7 = Gray7.scaled(self.div7.width(), self.div7.height())
                self.div7.setPixmap(QtGui.QPixmap.fromImage(Gray7))



    def connetSQL(self):
        conn = pymysql.connect(host='localhost',port=3306,user='root',password='root',db='opencv',charset='utf8')
        cur = conn.cursor()
        #获取游标
        return conn,cur



    def QImage2Pixmap(self,i,divs):
        return QtGui.QImage(divs[0][i].data, divs[0][i].shape[1], divs[0][i].shape[0],
                             QtGui.QImage.Format_Grayscale8)

    def slot_btn_function(self):
        self.hide()
        self.s = openCamera.CamShow()
        self.s.show()
    def show_preprocess(self):
        if self.Img_preprocess.any():
            cv2.imshow("preprocess",self.Img_preprocess)

    def sql_btn_function(self):
        self.s = openSearch.Searchwindow()
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
    #app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window = mywindow()
    window.show()
    sys.exit(app.exec_())