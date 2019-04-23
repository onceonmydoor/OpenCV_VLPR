

from PyQt5 import QtWidgets,QtGui
import sys
from test import Ui_Form #导入生成的界面类
from PyQt5.QtWidgets import QFileDialog
from predict import Predict
class mywindow(QtWidgets.QWidget,Ui_Form):
    def __init__(self):
        super(mywindow,self).__init__()
        self.setupUi(self)
        #定义槽函数
    def openimage(self):
    #打开文件路径
    #设置文件扩展名过滤，注意用双分号间隔
        imgPath , imgType = QFileDialog.getOpenFileName(self,"打开图片","","*.jpg;;*.png;;*.jpeg;;*.bmp;;All Files (*)")
        print(imgPath)
        #利用qlabel显示图片
        png = QtGui.QPixmap(imgPath).scaled(self.imgLabel.width(),self.imgLabel.height())
        self.imgLabel.setPixmap(png)


        ###################识别#####################
        q = Predict()
        # if q.isdark("test\\timg.jpg"):
        # print("是黑夜拍的")
        afterprocess, old = q.preprocess(imgPath)
        # cv2.namedWindow("yuchuli", cv2.WINDOW_NORMAL)
        # cv2.imshow("yuchuli", afterprocess)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        colors, card_imgs = q.locate_carPlate(afterprocess, old)
        # colors,card_imgs = q.img_only_color(old,afterprocess)
        result, roi, color = q.char_recogize(colors, card_imgs)  # all list
        if len(result) == 0:
            print("未能识别到车牌")
        else:
            for r in range(len(result)):
                print("车牌的颜色为：" + color[r])
                self.colorLabel.setText(color[r])
                print(result[r])

                self.NumLabel.setText(''.join(result[r]))
                print("\n")
        ###################识别#####################


if __name__ == '__main__':

    app =QtWidgets.QApplication(sys.argv)
    window = mywindow()
    window.show()
    sys.exit(app.exec_())