from camera import Ui_camera
import sys
from PyQt5.QtWidgets import QApplication,QMainWindow,QFileDialog
from PyQt5.QtCore import QTimer,QCoreApplication,QDateTime
import time
import cv2
import open
import qimage2ndarray
import threading
from PyQt5.QtGui import QPixmap
from predict import Predict
from PyQt5 import QtWidgets,QtGui,QtCore

class CamShow(QMainWindow,Ui_camera):
    def __del__(self):
        try:
            self.camera.release()  # 释放资源
        except:
            return

    def __init__(self,parent=None):
        super(CamShow,self).__init__(parent)
        thread_run = False
        self.setupUi(self)
        self.PreSliders()#保证两个控件的值始终相等
        self.PreWidgets()
        self.PrepParameters()#定义并初始化程序运行过程中用到的变量
        self.CallBackFunctions()
        self.Timer = QTimer()#计时器
        self.Timer.timeout.connect(self.TimerOutFun)#使用计时器实现对摄像头图像的循环读取和显示

        #当前时间
        self.timeLabel.setFixedWidth(200)
        self.timeLabel.setStyleSheet("QLabel{background:white;}"
                                      "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:宋体;}"
                                      )
        Ctimer = QTimer(self)
        Ctimer.timeout.connect(self.showCurrTime)
        Ctimer.start()

        self.Picbtn.clicked.connect(self.pic_model)




    def showCurrTime(self):
        datetime = QDateTime.currentDateTime()
        self.timeLabel.setText("     "+datetime.toString())




    def PreSliders(self):#保证两个控件的值始终相等
        self.RedColorSld.valueChanged.connect(self.RedColorSpB.setValue)
        self.RedColorSpB.valueChanged.connect(self.RedColorSld.setValue)
        self.GreenColorSld.valueChanged.connect(self.GreenColorSpB.setValue)
        self.GreenColorSpB.valueChanged.connect(self.GreenColorSld.setValue)
        self.BlueColorSld.valueChanged.connect(self.BlueColorSpB.setValue)
        self.BlueColorSpB.valueChanged.connect(self.BlueColorSld.setValue)
        self.ExpTimeSld.valueChanged.connect(self.ExpTimeSpB.setValue)
        self.ExpTimeSpB.valueChanged.connect(self.ExpTimeSld.setValue)
        self.GainSpB.valueChanged.connect(self.GainSld.setValue)
        self.GainSld.valueChanged.connect(self.GainSpB.setValue)
        self.BrightSld.valueChanged.connect(self.BrightSpB.setValue)
        self.BrightSpB.valueChanged.connect(self.BrightSld.setValue)
        self.ContrastSld.valueChanged.connect(self.ContrastSpB.setValue)
        self.ContrastSpB.valueChanged.connect(self.ContrastSld.setValue)

    def PreWidgets(self):#初始化控件
        self.PrepCamera()
        self.StopBt.setEnabled(False)
        self.RecordBt.setEnabled(False)
        self.RedColorSld.setEnabled(False)
        self.RedColorSpB.setEnabled(False)
        self.GreenColorSld.setEnabled(False)
        self.GreenColorSpB.setEnabled(False)
        self.BlueColorSld.setEnabled(False)
        self.BlueColorSpB.setEnabled(False)
        self.ExpTimeSld.setEnabled(False)
        self.ExpTimeSpB.setEnabled(False)
        self.GainSld.setEnabled(False)
        self.GainSpB.setEnabled(False)
        self.BrightSld.setEnabled(False)
        self.BrightSpB.setEnabled(False)
        self.ContrastSld.setEnabled(False)
        self.ContrastSpB.setEnabled(False)

    def PrepCamera(self):#开启摄像头
        try:
            self.camera = cv2.VideoCapture(0)
            self.MsgTE.clear()
            self.MsgTE.append('Oboard camera connected.')
        except Exception as e:
            self.MsgTE.clear()
            self.MsgTE.append(str(e))

    def PrepParameters(self):#参数的初始化
        self.RecordFlag = 0
        self.RecordPath = "D:/finalProjrct/OpenCV_VLPR/video"
        self.FilePathLE.setText(self.RecordPath)
        self.Image_num = 0
        self.R = 1
        self.G = 1
        self.B = 1

        self.ExpTimeSld.setValue(self.camera.get(15))
        self.SetExposure()
        self.GainSld.setValue(self.camera.get(14))
        self.SetGain()
        self.BrightSld.setValue(self.camera.get(10))
        self.SetBrightness()
        self.ContrastSld.setValue(self.camera.get(11))
        self.SetContrast()
        self.MsgTE.clear()

    def CallBackFunctions(self):#函数的调用
        self.FilePathBt.clicked.connect(self.SetFilePath)
        self.ShowBt.clicked.connect(self.StartCamera)
        self.StopBt.clicked.connect(self.StopCamera)
        self.RecordBt.clicked.connect(self.RecordCamera)
        self.ExitBt.clicked.connect(self.ExitApp)
        self.ExpTimeSld.valueChanged.connect(self.SetExposure)
        self.GainSld.valueChanged.connect(self.SetGain)
        self.BrightSld.valueChanged.connect(self.SetBrightness)
        self.ContrastSld.valueChanged.connect(self.SetContrast)
        self.RedColorSld.valueChanged.connect(self.SetR)
        self.GreenColorSld.valueChanged.connect(self.SetG)
        self.BlueColorSld.valueChanged.connect(self.SetB)

    def StartCamera(self):
        self.ShowBt.setEnabled(False)
        self.StopBt.setEnabled(True)
        self.RecordBt.setEnabled(True)
        self.RedColorSld.setEnabled(True)
        self.RedColorSpB.setEnabled(True)
        self.GreenColorSld.setEnabled(True)
        self.GreenColorSpB.setEnabled(True)
        self.BlueColorSld.setEnabled(True)
        self.BlueColorSpB.setEnabled(True)
        self.ExpTimeSld.setEnabled(True)
        self.ExpTimeSpB.setEnabled(True)
        self.GainSld.setEnabled(True)
        self.GainSpB.setEnabled(True)
        self.BrightSld.setEnabled(True)
        self.BrightSpB.setEnabled(True)
        self.ContrastSld.setEnabled(True)
        self.ContrastSpB.setEnabled(True)
        self.RecordBt.setText('录像')

        # 开启识别线程
        p = threading.Thread(target=self.vedio_thread)
        p.setDaemon(True)#子线程附加在主线程上
        p.start()


        self.Timer.start(1)
        self.timelb = time.time()



    def StopCamera(self):
        if self.StopBt.text() == '暂停':
            self.StopBt.setText('继续')
            self.RecordBt.setText('保存')
            self.Timer.stop()
        elif self.StopBt.text() == '继续':
            self.StopBt.setText('暂停')
            self.RecordBt.setText('录像')
            self.Timer.start(1)

    def TimerOutFun(self):
        success, img = self.camera.read()
        if success:
            self.Image = self.ColorAdjust(img)
            self.DispImg()
            self.Image_num += 1
            if self.RecordFlag:
                self.video_writer.write(img)
            if self.Image_num % 10 == 9:
                frame_rate = 10 / (time.time() - self.timelb)
                self.FmRateLCD.display(frame_rate)
                self.timelb = time.time()
                # size=img.shape
                self.ImgWidthLCD.display(self.camera.get(3))
                self.ImgHeightLCD.display(self.camera.get(4))
        else:
            self.MsgTE.clear()
            self.MsgTE.setPlainText('Image obtaining failed.')

    def DispImg(self):#显示出图像
        img = cv2.cvtColor(self.Image,cv2.COLOR_BGR2RGB)
        qimg = qimage2ndarray.array2qimage(img)
        self.DispLb.setPixmap(QPixmap(qimg))
        self.DispLb.show()



    def SetR(self):
        R = self.RedColorSld.value()
        self.R = R/255

    def SetG(self):
        G = self.GreenColorSld.value()
        self.G = G/255

    def SetB(self):
        B = self.BlueColorSld.value()
        self.B = B/255


    def ColorAdjust(self,img):#读显示图像的RGB调整
        try:
            B = img[:,:,0]
            G = img[:,:,1]
            R = img[:,:,2]
            B = B*self.B
            G = G*self.G
            R = R*self.R
            Temp_img = img
            Temp_img[:,:,0] = B
            Temp_img[:,:,1] = G
            Temp_img[:,:,2] = R
            return Temp_img
        except Exception as e:
            self.MsgTE.setPlainText("出错啦！"+str(e))


    def SetExposure(self):
        exposure_time_toset = self.ExpTimeSld.value()
        try:
            self.camera.set(15,exposure_time_toset)
            self.MsgTE.setPlainText("The exposure time is set to "+str(self.camera.get(15)))
        except Exception as e:
            self.MsgTE.setPlainText("出错啦！"+str(e))

    def SetGain(self):
        gain_toset = self.GainSld.value()
        try:
            self.camera.set(14,gain_toset)
            self.MsgTE.setPlainText("The gain is set to "+str(self.camera.get(14)))
        except Exception as e:
            self.MsgTE.setPlainText("出错啦！"+str(e))


    def SetBrightness(self):
        brightness_toset = self.BrightSld.value()
        try:
            self.camera.set(10,brightness_toset)
            self.MsgTE.setPlainText("The brightness is set to "+str(self.camera.get(10)))
        except Exception as e:
            self.MsgTE.setPlainText("出错啦！"+str(e))

    def SetContrast(self):
        contrast_toset = self.ContrastSld.value()
        try:
            self.camera.set(11,contrast_toset)
            self.MsgTE.setPlainText("The contrast is set to "+str(self.camera.get(11)))
        except Exception as e:
            self.MsgTE.setPlainText("出错啦！"+str(e))

    def SetFilePath(self):
        dirname = QFileDialog.getExistingDirectory(self,"浏览",".")
        if dirname:
            self.FilePathLE.setText(dirname)
            self.RecordPath = dirname +"/"


    def RecordCamera(self):
        tag = self.RecordBt.text()
        if tag == "保存":
            try:
                image_name = self.RecordPath +"image" +time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))+'.jpg'
                print(image_name)
                cv2.imwrite(image_name, self.Image)
                self.MsgTE.clear()
                self.MsgTE.setPlainText('Image saved.')
            except Exception as e:
                self.MsgTE.clear()
                self.MsgTE.setPlainText(str(e))
        elif tag=='录像':
            self.RecordBt.setText('停止')
            video_name = self.RecordPath + 'video' + time.strftime('%Y%m%d%H%M%S',time.localtime(time.time())) + '.avi'
            fps = self.FmRateLCD.value()
            size = (self.Image.shape[1],self.Image.shape[0])
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            self.video_writer = cv2.VideoWriter(video_name, fourcc,self.camera.get(5), size)
            self.RecordFlag=1
            self.MsgTE.setPlainText('Video recording...')
            self.StopBt.setEnabled(False)
            self.ExitBt.setEnabled(False)
        elif tag == '停止':
            self.RecordBt.setText('录像')
            self.video_writer.release()
            self.RecordFlag = 0
            self.MsgTE.setPlainText('Video saved.')
            self.StopBt.setEnabled(True)
            self.ExitBt.setEnabled(True)

    def pic_model(self):
        self.hide()
        self.s = open.mywindow()
        self.s.show()



    def vedio_thread(self):
        self.thread_run = True
        predict_time = time.time()
        while self.thread_run:
            _ , img_bgr = self.camera.read()
            if time.time() - predict_time >2:
                self.predict_frame(img_bgr)
                predict_time = time.time()

    def predict_frame(self,imgPath):
        Eng2Chi = {"green": "绿色", "blue": "蓝色", "yellow": "黄色"}
        q = Predict()
        afterprocess, old = q.preprocess(imgPath)
        colors, card_imgs = q.locate_carPlate(afterprocess, old)
        result, roi, color, divs = q.char_recogize(colors, card_imgs)  # all list
        if len(result) == 0:
            print("未能识别到车牌")
            self.colorLabel.setText("抱歉未能识别到车牌")
            self.NumLabel.setText("抱歉，未能识别到车牌")
        else:
            for r in range(len(result)):
                print("#" * 10 + "识别结果是" + "#" * 10)
                print("车牌的颜色为：" + Eng2Chi[color[r]])
                self.colorLabel.setText(Eng2Chi[color[r]])
                print(result[r])
                result[r].insert(2, "-")
                self.NumLabel.setText(''.join(result[r]))
                print("#" * 25)
                # roi[r] = cv2.cvtColor(roi[r], cv2.COLOR_BGR2RGB)
                # QtImg = QtGui.QImage(roi[r].data, roi[r].shape[1], roi[r].shape[0], QtGui.QImage.Format_RGB888)
                # # 显示图片到label中
                # self.location.resize(QtCore.QSize(roi[r].shape[1], roi[r].shape[0]))
                # self.location.setPixmap(QtGui.QPixmap.fromImage(QtImg))

                print("\n")



    def ExitApp(self):
        self.Timer.stop()#停止计时器
        self.camera.release()#释放摄像头
        self.MsgTE.setPlainText("Exiting the application..")

        QCoreApplication,quit()






if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = CamShow()
    ui.show()
    sys.exit(app.exec_())
