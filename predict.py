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
MIN_AREA = 1000 #车牌区域的允许最大面积（不能过大）
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


    def grey_scale(self,image):
        img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        cv2.imshow("灰度", img_gray)
        cv2.waitKey()
        cv2.destroyAllWindows()
        rows,cols = img_gray.shape
        flat_gray = img_gray.reshape((cols*rows)).tolist()

        A = min(flat_gray)
        B = max(flat_gray)
        print('A = %d,B = %d' %(A,B))
        output = np.uint8(255 / (B - A)*(img_gray - A) + 0.5)
        return output




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
            print("图片长为{}，图片宽为{}".format(pic_hight,pic_width))
            #适当缩小图片
            if pic_width > MAX_WIDTH:
                resize_rate = MAX_WIDTH / pic_width
                img = cv2.resize(img, (MAX_WIDTH, int(pic_hight * resize_rate)), interpolation=cv2.INTER_AREA)
            
            gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#转成灰度图

            blur_img = cv2.blur(gray_img,(3,3))#均值模糊

            sobel_img = cv2.Sobel(blur_img,cv2.CV_16S, 1, 0, ksize=3)#sobel获取垂直边缘
            sobel_img = cv2.convertScaleAbs(sobel_img)

            hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)#转成HSV

            # cv2.imshow("hsv",hsv_img)
            # cv2.waitKey(0)

            h , s , v = hsv_img[:, :, 0], hsv_img[:, :, 1],hsv_img[:, :, 2]

            #黄色的色调区间再[26,34],蓝色的色调区间再[100,124]，绿色的色调区间在[35,100]
            blue_img = (((h > 15) & (h <= 34)) | ((h > 35) & (h < 100)) | ((h >= 100) & (h <= 124))) & (s > 70) & (v > 70)
            blue_img = blue_img.astype('float32')

            mix_img = np.multiply(sobel_img, blue_img)
            #cv2.imshow('mix', mix_img)

            mix_img = mix_img.astype(np.uint8)

            ret, binary_img = cv2.threshold(mix_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            #cv2.imshow('binary',binary_img)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(21,5))
            close_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
            

            return close_img , img



         
            
            
            #return img_edge2, oldimg

            ##生成预处理图像，车牌识别的第一步
    
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
            if count > int(col_num_limit):
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

    def locate_carPlate(self,img_coutours,oldimg):
        """
        :param img_contours :预处理好的图像
        :param oldimg: 原图像
        :return: 已经定位好的车牌
        :此方法为确定车牌的定位，通过矩形长宽比和颜色定位图像t
        """
        # if img_coutours.any():
        #     config.set_name(img_coutours)
        pic_width,pic_hight = oldimg.shape[:2]
        car_contours = []
        card_contours = self.img_findContours(img_coutours,oldimg)

        print("精确定位中...")
        print("正在调整车牌位置...")
        card_imgs = self.img_Transform(card_contours,oldimg)
        #开始使用颜色车牌定位，排除不是车牌的矩形，目前只识别蓝、绿、黄三种常规颜色的车牌
        
        print("正在根据颜色再定位...")
        colors , card_imgs = self.img_color(card_imgs)
            
        #返回可能存在的
        for i in colors[:]:
            if i == "no":
                index = colors.index(i)
                colors.remove("no")
                card_imgs.pop(index)
        #show图片
        for card_img in card_imgs:
            if card_img.any():
                cv2.imshow("dingwei", card_img)
                cv2.waitKey()
                cv2.destroyAllWindows()
        
        return colors,card_imgs  #可能存在多个车牌，暂时保留列表结构                                                                          


    def img_only_color(self, oldimg, img_contours):
        """
        :param filename: 图像文件
        :param oldimg: 原图像文件
        :return: 已经定位好的车牌
        """
        pic_hight, pic_width = img_contours.shape[:2]
        lower_blue = np.array([100, 110, 110])
        upper_blue = np.array([130, 255, 255])
        lower_yellow = np.array([15, 55, 55])
        upper_yellow = np.array([50, 255, 255])
        lower_green = np.array([50, 50, 50])
        upper_green = np.array([100, 255, 255])
        hsv = cv2.cvtColor(oldimg, cv2.COLOR_BGR2HSV)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_green = cv2.inRange(hsv, lower_yellow, upper_green)
        output = cv2.bitwise_and(hsv, hsv, mask=mask_blue + mask_yellow + mask_green)
        # 根据阈值找到对应颜色

        output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        Matrix = np.ones((20, 20), np.uint8)
        img_edge1 = cv2.morphologyEx(output, cv2.MORPH_CLOSE, Matrix)
        img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, Matrix)

        card_contours = self.img_findContours(img_edge2,oldimg)
        card_imgs = self.img_Transform(card_contours, oldimg)
        colors, car_imgs = self.img_color(card_imgs)
        return colors,car_imgs



    def char_recogize(self,colors,card_imgs):
        #车牌字符识别
        predict_result = []
        roi = None
        card_color = None
        for i , color in enumerate(colors):   
            if color in ("blue","yellow","green"):
                card_img = card_imgs[i]
                gray_img = cv2.cvtColor(card_img,cv2.COLOR_BGR2GRAY)#转成灰度图
                
                #黄、绿车牌字符比背景暗、与蓝车牌刚好相反，所以黄、绿车牌需要反向
                if color == "green" or color == "yellow":
                    gray_img = cv2.bitwise_not(gray_img)
                ret , gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)#OTSU  ,字符显示的第一步
                #（灰度图，阈值，最大值，阈值类型）把阈值设为0，算法会找到最优阈值
                cv2.imshow("erzhihua", gray_img)
                cv2.waitKey()
                cv2.destroyAllWindows()
                
    def img_findContours(self,img_coutours,oldimg):
        pic_hight, pic_width = oldimg.shape[:2]#获取图片的长宽
        #查找图像边缘整体形成的矩形区域，可能有很多，车牌就在其中一个矩形
        image,contours,hierarchy = cv2.findContours(img_coutours,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt)>MIN_AREA]#选择出大于最小的矩形的可能车牌区域
        print('len(contours)',len(contours))
        if len(contours)==0:
            print("没有找到可能是车牌的区域")
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
                cv2.imshow("edge4", oldimg)
                cv2.waitKey()
                cv2.destroyAllWindows()
                print(rect)
        print(len(car_contours))#有几个矩形
        return car_contours
                
    def img_Transform(self,card_contours,oldimg):
        rect_h, rect_w = oldimg.shape[:2]#获取图片的长宽
        card_imgs = []
        #矩形区域可能是倾斜的矩阵，需要矫正，以便使用颜色定位
        for rect in card_contours:#rect((中心点坐标)，（宽，长），角度)
            return_flag = False
            angle = rect[2]#获得矩形旋转的角度
            print("角度是{}".format(angle))
            print("宽是{},长是{}".format(rect[1][0],rect[0][1]))

            rect = (rect[0],(rect[1][0] + 5,rect[1][1] + 5),angle) #扩大范围，避免车牌的边缘被排除

            if angle == 0:
                return_flag = True
            if angle == -90 and rect_w<rect_h:
                rect_w , rect_h = rect_h , rect_w
                return_flag = True
            if return_flag:
                car_img = oldimg[int(rect[0][1]-rect_h/2):int(rect[0][1]+rect_h/2),int(rect[0][0]-rect_w/2):int(rect[0][0]+rect_w/2)]
                return car_img

            box = cv2.boxPoints(rect)
            height_point = right_point = [0,0]#设定右上是0，0
            left_point = low_point = [rect[0][0],rect[0][1]]
            for point in box:
                if left_point[0] > point[0]:
                    left_point = point
                if low_point[1] > point[1]:#最低点的Y大于现在的y
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
                dst = cv2.warpAffine(oldimg,M,(round(rect_w*2),round(rect_h*2)))#仿射变换,OUTPUT Array，输出图像
                img_math.point_limit(new_right_point)
                img_math.point_limit(left_point)
                img_math.point_limit(height_point)
                card_img = dst[int(left_point[1]):int(height_point[1]),int(left_point[0]):int(new_right_point[0])]#摆正图像
                #show
                card_imgs.append(card_img)
                cv2.imshow("card2",card_img)
                cv2.waitKey(0)
            elif left_point[1] > right_point[1]:  #负角度
                new_left_point = [left_point[0],height_point[1]]
                pts2 = np.float32([new_left_point,height_point,right_point])  #字符只是高度需要改变
                pts1 = np.float32([left_point,height_point,right_point])
                M = cv2.getAffineTransform(pts1,pts2)
                dst = cv2.warpAffine(oldimg,M,(round(rect_w*2),round(rect_h*2)))
                img_math.point_limit(right_point)
                img_math.point_limit(height_point)
                img_math.point_limit(new_left_point)
                card_img = dst[int(right_point[1]):int(height_point[1]),int(new_left_point[0]):int(right_point[0])]
                #show
                card_imgs.append(card_img)
                cv2.imshow("card2",card_img)
                cv2.waitKey(0)
        return card_imgs

    def img_color(self,card_imgs):
        colors = []
        print("可能存在{}个车牌".format(len(card_imgs)))
        for card_index, card_img in enumerate(card_imgs):
            green = yellow = blue = black = white = 0
            if card_img.any():
                card_img_hsv = cv2.cvtColor(card_img,cv2.COLOR_BGR2HSV)#色相、饱和度、明度
                #TODO：可能会存在转换失败的问题，原因来自于矫正矩形失败
                if card_img_hsv is None:
                    continue
                cv2.imshow("hsv",card_img_hsv)
                cv2.waitKey(0)

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
                elif blue * 2.3 >= card_img_count:
                    color = "blue"
                    limit1 = 100
                    limit2 = 124 #存在蓝色偏紫的情况
                elif black + white >= int(card_img_count*0.7):
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

            return colors , card_imgs

if __name__ == '__main__':
    q = Predict()
    afterprocess,old = q.preprocess("test\\9.png")
    #afterprocess,old=q.preprocess("test\\Yes_img\\3_2.jpg")
    cv2.imshow("预处理", afterprocess)
    cv2.waitKey()
    cv2.destroyAllWindows()
    colors,card_imgs=q.locate_carPlate(afterprocess,old)
    #colors,card_imgs = q.img_only_color(old,afterprocess)
    q.char_recogize(colors,card_imgs)
