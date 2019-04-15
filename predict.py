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
    
    def accurate_place(self, card_img_hsv, limit1, limit2, color):#微调车牌位置
        row_num, col_num = card_img_hsv.shape[:2]
        xl = col_num
        xr = 0
        yh = 0
        yl = row_num
        # col_num_limit = self.cfg["col_num_limit"]
        row_num_limit = self.cfg["row_num_limit"]
        col_num_limit = col_num * 0.8 if color != "green" else col_num * 0.5  # 绿色有渐变
        for i in range(row_num):
            count = 0
            for j in range(col_num):
                H = card_img_hsv.item(i, j, 0)
                S = card_img_hsv.item(i, j, 1)
                V = card_img_hsv.item(i, j, 2)
                if limit1 < H <= limit2 and 34 < S and 46 < V:
                    count += 1
            if count > col_num_limit:
                if yl > i:
                    yl = i
                if yh < i:
                    yh = i
        for j in range(col_num):
            count = 0
            for i in range(row_num):
                H = card_img_hsv.item(i, j, 0)
                S = card_img_hsv.item(i, j, 1)
                V = card_img_hsv.item(i, j, 2)
                if limit1 < H <= limit2 and 34 < S and 46 < V:
                    count += 1
            if count > row_num - row_num_limit:
                if xl > j:
                    xl = j
                if xr < j:
                    xr = j
        return xl, xr, yh, yl

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
                #oldimg = cv2.drawContours(oldimg, [box], 0, (0, 0, 255), 2)#在原图像上画出矩形
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
                #cv2.imshow("card",card_img)
                #cv2.waitKey(0)
            elif left_point[1] > right_point[1]:  #负角度
                new_left_point = [left_point[0],height_point[1]]
                pts2 = np.float32([new_left_point,height_point,right_point])  #字符只是高度需要改变
                pts1 = np.float32([left_point,height_point,right_point])
                M = cv2.warpAffine(oldimg,M,(pic_width,pic_hight))
                img_math.point_limit(right_point)
                img_math.point_limit(height_point)
                img_math.point_limit(new_left_point)
                card_img = dst[int(right_point[1]):int(height_point[1]),int(new_left_point[0]):int(right_point[0])]
                card_imgs.append(card_img)
                #cv2.imshow("card2",card_img)
                cv2.waitKey(0)
        #开始使用颜色车牌定位，排除不是车牌的矩形，目前只识别蓝、绿、黄三种常规颜色的车牌
        colors = []
        for card_index, card_img in enumerate(card_imgs):
            green = yellow = blue = black = white = 0
            card_img_hsv = cv2.cvtColor(card_img,cv2.COLOR_BGR2HSV)#色相、饱和度、明度
            #TODO：可能会存在转换失败的问题，原因来自于矫正矩形失败
            if card_img_hsv is None:
                continue
            row_num , col_num = card_img_hsv.shape[:2]#获取长宽
            card_img_count = row_num * col_num
            for i in range(row_num): #O(N^2)
                for j in range(col_num):
                    H = card_img_hsv.item(i,j,0)
                    S = card_img_hsv.item(i,j,1)
                    V = card_img_hsv.item(i,j,2)
                    if 11 < H <= 34 and  S > 34:#图片分辨率调整
                        yellow += 1
                    elif 35 < H <= 99 and S > 34:
                        green += 1
                    elif 99 < H <= 124 and S > 34: 
                        blue += 1
                    if 0 < H < 180 and 0 < S < 255 and 0 < V < 46:
                        black += 1
                    elif 0 < H < 180 and 0 < S < 43 and 221 < V < 225: 
                        white += 1
            color = "no" #默认矩形不存在车牌
            limit1 = limit2 = 0
            if yellow * 2 >= card_img_count:#黄色大于50%
                color = "yellow"
                limit1 = 11
                limit2 = 34 # 存在黄色偏绿的情况
            elif green * 2 >= card_img_count:#绿色大于50%
                color = "green"
                limit1 = 35
                limit2 = 99 #存在绿色偏蓝的情况
            elif blue * 2 >= card_img_count:
                color = "blue"
                limit1 = 100
                limit2 = 124 #存在蓝色偏紫的情况
            elif black + white >= (card_img_count*0.7):
                color = "bw"
            print(color)
            colors.append(color)
            print("blue:{},green:{},yellow:{},black:{},white:{},count:{}".format(blue,green,yellow,black,white,card_img_count))


            #根据车牌颜色在定位，缩小非车牌的边界区域
            xl, xr, yh, yl = self.accurate_place(card_img_hsv, limit1, limit2, color)
            if yl == yh and xl == xr:
                continue
            need_accurate = False
            if yl >= yh:
                yl = 0
                yh = row_num
                need_accurate = True
            if xl >= xr:
                xl = 0
                xr = col_num
                need_accurate = True
            card_imgs[card_index] = card_img[yl:yh, xl:xr] if color !="green" or yl < (yh - yl) // 4 else card_img[yl - (yh - yl) // 4:yh,xl:xr] 


            if need_accurate: #可能x或y方向未缩小，需要再试一次
                card_img = card_imgs[card_index]
                card_img_hsv = cv2.cvtColor(card_img,cv2.COLOR_BGR2HSV)
                xl, xr, yh, yl = self.accurate_place(card_img_hsv, limit1, limit2, color)
                if yl == yh and xl == xr:
                    continue
                if yl >= yh:
                    yl = 0
                    yh = row_num
                if xl >= xr:
                    xl = 0
                    xr = col_num
            card_imgs[card_index] = card_img[yl:yh, xl:xr] if color != "green" or yl < (yh - yl) // 4 else card_img[yl - (yh - yl) // 4:yh,xl:xr]
            
        #返回可能存在的
        for i in colors[:]:
            if i == "no":
                index = colors.index(i)
                colors.remove("no")
                card_imgs.pop(index)
        #show图片
        for card_img in card_imgs:
            cv2.imshow("dingwei", card_img)
            cv2.waitKey()
            cv2.destroyAllWindows()
        
        return colors,card_imgs  #可能存在多个车牌，暂时保留列表结构                                                                          


    #def char_recogize(colors,card_imgs):
        #车牌字符识别
   
        

if __name__ == '__main__':
    q = Predict()
    afterprocess,old=q.preprocess("car4.jpg")
    q.img_color_contours(afterprocess,old)
