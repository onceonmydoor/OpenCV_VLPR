"""
车牌识别的主要方法
"""
import os
import cv2
import numpy as np
import debug
import img_math
import img_rec
import Train_SVM
import config
import json

SZ = 20 #训练图片长宽 (训练样本是20*20的图片)
MAX_WIDTH = 2000 #原始图片最大宽度
MIN_AREA = 2000 #车牌区域的允许最大面积（不能过大）
PROVINCE_START = 1000 #用于字符分类


class Predict:
    def __init__(self):
        # 车牌识别的部分参数保存在js中，便于根据图片分辨率做调整
        f = open('config.js')
        j = json.load(f)
        for c in j["config"]:
            if c["open"]:
                self.cfg = c.copy()
                break
            else:
                raise RuntimeError('没有设置有效配置参数')
    def __del__(self):
        pass

    def preprocess(self, car_pic_file):
            """
            :param car_pic_file: 图像文件
            :return:已经处理好的图像文件 原图像文件
            """
            if type(car_pic_file) == type(""):
                img = img_math.img_read(car_pic_file)
            else:
                img = car_pic_file

            pic_hight, pic_width = img.shape[:2]
            if pic_width > MAX_WIDTH:
                resize_rate = MAX_WIDTH / pic_width
                img = cv2.resize(img, (MAX_WIDTH, int(pic_hight * resize_rate)), interpolation=cv2.INTER_AREA)
            # 缩小图片

            blur = self.cfg["blur"]
            # 高斯去噪
            if blur > 0:
                img = cv2.GaussianBlur(img, (blur, blur), 0)
            #调整图片分辨率
            oldimg = img
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 转化成灰度图像

            Matrix = np.ones((20, 20), np.uint8)
            img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, Matrix)
            img_opening = cv2.addWeighted(img, 1, img_opening, -1, 0)
            # 创建20*20的元素为1的矩阵 开操作，并和img重合,除去孤立的小点，毛刺和小桥，而总的位置和形状不变

            ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img_edge = cv2.Canny(img_thresh, 100, 200)#Canny算子
            # Otsu’s二值化 找到图像边缘


            
            Matrix = np.ones((4, 19), np.uint8)
            img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, Matrix)
            img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, Matrix)
            #使用开运算和闭运算让图像边缘成为一个整体
            
            return img_edge2, oldimg

            ###生成预处理图像，车牌识别的第一步



    def img_color_contours(self,img_coutours,oldimg):
        """
        :param img_contours :预处理好的图像
        :param oldimg: 原图像
        :return: 已经定位好的车牌
        :此方法为确定车牌的定位，通过矩形长宽比和颜色定位图像
        """
        # if img_coutours.any():
        #     config.set_name(img_coutours)
        pic_hight, pic_width = oldimg.shape[:2]#获取图片的长宽
        #查找图像边缘整体形成的矩形区域，可能有很多，车牌就在其中一个矩形
        image,contours,hierarchy = cv2.findContours(img_coutours,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt)>MIN_AREA]#选择出大于最小的矩形的可能车牌区域
        print('len(contours)',len(contours))
        #一一排除不是车牌的矩形区域
        car_contours = []
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)# 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
            area_width,area_height = rect[1]
            if area_width < area_height:
                area_width , area_height = area_height , area_width
            wh_ratio = area_width / area_height #长宽比
            print(wh_ratio)
            #要求矩形区域长宽比在2到5.5之间，2到5.5是车牌的长宽比，其余的矩形排除
            if wh_ratio > 2 and wh_ratio < 5.5:
                car_contours.append(rect)
                box = cv2.boxPoints(rect)
                box = np.int0(box)#int0==int64
                oldimg = cv2.drawContours(oldimg, [box], 0, (0, 0, 255), 2)#在原图像上画出矩形
                #cv2.imshow("edge4", oldimg)
                #cv2.waitKey()
                #cv2.destroyAllWindows()
                print(rect)

        print(len(car_contours))#有几个矩形

        print("精确定位...")
        card_imgs = []
        #矩形区域可能是倾斜的矩阵，需要矫正，以便使用颜色定位
        for rect in car_contours:
            if rect[2] > -1 and rect[2] < 1:
                angle = 1
            else:
                angle = rect[2]#获得矩形旋转的角度
            rect = (rect[0],(rect[1][0] + 5,rect[1][1] + 5),angle) #扩大范围，避免车牌的边缘被排除

            box = cv2.boxPoints(rect)
            height_point = right_point = [0,0]#设定右上是0，0
            left_point = low_point = [pic_width,pic_hight]
            for point in box:
                if left_point[0] > point[0]:
                    left_point = point
                if low_point[1] > point[1]:
                    low_point = point 
                if height_point[1] < point[1]:
                    height_point = point
                if right_point[0] < point[0]:
                    right_point = point
            
            if left_point[1] <= right_point[1]: #正角度
                new_right_point = [right_point[0],height_point[1]]
                pts2 = np.float32([left_point,height_point,new_right_point])
                pts1 = np.float32([left_point,height_point,right_point])
                M = cv2.getAffineTransform(pts1,pts2) #INPUT Array 2*3的变换矩阵
                dst = cv2.warpAffine(oldimg,M,(pic_width,pic_hight))#仿射变换,OUTPUT Array，输出图像
                img_math.point_limit(new_right_point)
                img_math.point_limit(left_point)
                img_math.point_limit(height_point)
                card_img = dst[int(left_point[1]):int(height_point[1]),int(left_point[0]):int(new_right_point[0])]#摆正图像
                card_imgs.append(card_img)
                cv2.imshow("card",card_img)
                cv2.waitKey(0)







if __name__ == '__main__':
    q = Predict()
    afterprocess,old=q.preprocess("car4.jpg")
    q.img_color_contours(afterprocess,old)
