"""
车牌识别的主要方法
"""
import os
import cv2
import numpy as np
import img_math
import Train_SVM
import json
from PIL import ImageStat
from PIL import Image
import time



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

    def verify_color(self, rotate_rect, src_image):
        img_h, img_w = src_image.shape[:2]
        mask = np.zeros(shape=[img_h + 2, img_w + 2], dtype=np.uint8)
        connectivity = 4  # 种子点上下左右4邻域与种子颜色值在[loDiff,upDiff]的被涂成new_value，也可设置8邻域
        loDiff, upDiff = 30, 30
        new_value = 255
        flags = connectivity
        flags |= cv2.FLOODFILL_FIXED_RANGE  # 考虑当前像素与种子象素之间的差，不设置的话则和邻域像素比较
        flags |= new_value << 8
        flags |= cv2.FLOODFILL_MASK_ONLY  # 设置这个标识符则不会去填充改变原始图像，而是去填充掩模图像（mask）

        rand_seed_num = 5000  # 生成多个随机种子
        valid_seed_num = 200  # 从rand_seed_num中随机挑选valid_seed_num个有效种子
        adjust_param = 0.1
        box_points = cv2.boxPoints(rotate_rect)
        box_points_x = [n[0] for n in box_points]
        box_points_x.sort(reverse=False)
        adjust_x = int((box_points_x[2] - box_points_x[1]) * adjust_param)
        col_range = [box_points_x[1] + adjust_x, box_points_x[2] - adjust_x]
        box_points_y = [n[1] for n in box_points]
        box_points_y.sort(reverse=False)
        adjust_y = int((box_points_y[2] - box_points_y[1]) * adjust_param)
        row_range = [box_points_y[1] + adjust_y, box_points_y[2] - adjust_y]
        # 如果以上方法种子点在水平或垂直方向可移动的范围很小，则采用旋转矩阵对角线来设置随机种子点
        if (col_range[1] - col_range[0]) / (box_points_x[3] - box_points_x[0]) < 0.4 \
                or (row_range[1] - row_range[0]) / (box_points_y[3] - box_points_y[0]) < 0.4:
            points_row = []
            points_col = []
            for i in range(2):
                pt1, pt2 = box_points[i], box_points[i + 2]
                x_adjust, y_adjust = int(adjust_param * (abs(pt1[0] - pt2[0]))), int(
                    adjust_param * (abs(pt1[1] - pt2[1])))
                if (pt1[0] <= pt2[0]):
                    pt1[0], pt2[0] = pt1[0] + x_adjust, pt2[0] - x_adjust
                else:
                    pt1[0], pt2[0] = pt1[0] - x_adjust, pt2[0] + x_adjust
                if (pt1[1] <= pt2[1]):
                    pt1[1], pt2[1] = pt1[1] + adjust_y, pt2[1] - adjust_y
                else:
                    pt1[1], pt2[1] = pt1[1] - y_adjust, pt2[1] + y_adjust
                temp_list_x = [int(x) for x in np.linspace(pt1[0], pt2[0], int(rand_seed_num / 2))]
                temp_list_y = [int(y) for y in np.linspace(pt1[1], pt2[1], int(rand_seed_num / 2))]
                points_col.extend(temp_list_x)
                points_row.extend(temp_list_y)
        else:
            points_row = np.random.randint(row_range[0], row_range[1], size=rand_seed_num)
            points_col = np.linspace(col_range[0], col_range[1], num=rand_seed_num).astype(np.int)

        points_row = np.array(points_row)
        points_col = np.array(points_col)
        hsv_img = cv2.cvtColor(src_image, cv2.COLOR_BGR2HSV)
        h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
        # 将随机生成的多个种子依次做漫水填充,理想情况是整个车牌被填充
        flood_img = src_image.copy()
        seed_cnt = 0
        for i in range(rand_seed_num):
            rand_index = np.random.choice(rand_seed_num, 1, replace=False)
            row, col = points_row[rand_index], points_col[rand_index]
            # 限制随机种子必须是车牌背景色
            if ((h[row, col] > 26) & (h[row, col] < 124)) & (s[row, col] > 70) & (v[row, col] > 70):
                cv2.floodFill(src_image, mask, (col, row), (255, 255, 255), (loDiff,) * 3, (upDiff,) * 3, flags)
                cv2.circle(flood_img, center=(col, row), radius=2, color=(0, 0, 255), thickness=2)
                seed_cnt += 1
                if seed_cnt >= valid_seed_num:
                    break
                    # ======================调试用======================#
                    # show_seed = np.random.uniform(1,100,1).astype(np.uint16)
                    # cv2.imshow('floodfill'+str(show_seed),flood_img)
                    # cv2.imshow('flood_mask'+str(show_seed),mask)
                    # ======================调试用======================#
                    # 获取掩模上被填充点的像素点，并求点集的最小外接矩形
        mask_points = []
        for row in range(1, img_h + 1):
            for col in range(1, img_w + 1):
                if mask[row, col] != 0:
                    mask_points.append((col - 1, row - 1))
        mask_rotateRect = cv2.minAreaRect(np.array(mask_points))
        return True, mask_rotateRect

    def grey_scale(self,image):
        img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # cv2.imshow("灰度", img_gray)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        rows,cols = img_gray.shape
        flat_gray = img_gray.reshape((cols*rows)).tolist()

        A = min(flat_gray)
        B = max(flat_gray)
        print('A = %d,B = %d' %(A,B))
        output = np.uint8(255 / (B - A)*(img_gray - A) + 0.5)
        return output

    def brightness1( car_pic ):
        im = Image.open(car_pic).convert('L')
        stat = ImageStat.Stat(im)   
        return stat.mean[0]
        print(brightness1('c:\\meiping1.png'))

    def isdark(self,car_pic):

        if type(car_pic) == type(""):
            img = img_math.img_read(car_pic)
        else:
            img = car_pic
        #把图片转换成为灰度图
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        #获取灰度图矩阵的行数和列数
        rows , cols = gray_img.shape[:2]
        dark_sum = 0
        dark_prop = 0
        piexs_sum = rows*cols

        #遍历灰度图的所有像素
        for row in gray_img:
            for col in row:
                if col<40:
                    dark_sum+=1
        dark_prop = dark_sum/(piexs_sum)
        print("总的黑色像素为"+str(dark_sum))
        print("总像素是："+str(piexs_sum))
        if dark_prop >= 0.75:
            return True
        return False

    #预处理
    def preprocess(self, car_pic_file):
            """
            :param car_pic_file: 图像文件
            :return:已经处理好的图像文件 原图像文件
            """

            if type(car_pic_file) == type(""):
                img = img_math.img_read(car_pic_file)
            else:
                img = car_pic_file

            if img.any() == None:
                return

            pic_hight, pic_width = img.shape[:2]
            print("图片长高为{}，图片长为{}".format(pic_hight,pic_width))
            #适当缩小图片
            if pic_width > MAX_WIDTH:#如果过大了
                resize_rate = MAX_WIDTH / pic_width
                img = cv2.resize(img, (MAX_WIDTH, int(pic_hight * resize_rate)), interpolation=cv2.INTER_AREA)

            #img = cv2.equalizeHist(img)
            
            gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#转成灰度图
            # cv2.imshow("gray", gray_img)
            # cv2.waitKey(0)

            # dst = cv2.equalizeHist(gray_img)
            # cv2.imshow("dst",dst)
            # cv2.waitKey(0)

            blur_img = cv2.blur(gray_img,(3,3))#均值模糊
            #blur_img = cv2.medianBlur(gray_img,3)
            #cv2.imshow("blur",blur_img)
            #cv2.waitKey(0)

            sobel_img = cv2.Sobel(blur_img,cv2.CV_16S, 1, 0, ksize=3)#sobel获取垂直边缘
            sobel_img = cv2.convertScaleAbs(sobel_img)

            #sobel_img = cv2.Canny(blur_img,100,200)
            cv2.imshow("sobel", sobel_img)
            cv2.waitKey(0)

            hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)#转成HSV

            # cv2.imshow("hsv",hsv_img)
            # cv2.waitKey(0)

            h , s , v = hsv_img[:, :, 0], hsv_img[:, :, 1],hsv_img[:, :, 2]

            #黄色的色调区间再[26,34],蓝色的色调区间再[100,124]，绿色的色调区间在[35,100]
            blue_img = (((h > 15) & (h <= 124))) & (s > 70) & (v > 70)#橙色和紫色
            blue_img = blue_img.astype('float32')

            mix_img = np.multiply(sobel_img, blue_img)
            # cv2.imshow('mix', mix_img)
            # cv2.waitKey(0)

            mix_img = mix_img.astype(np.uint8)

            ret, binary_img = cv2.threshold(mix_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # cv2.imshow('binary',binary_img)
            # cv2.waitKey(0)


            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(21,5))
            close_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

            ##config##
            threshold_m = int(pic_width/80)#根据自己调参出来的
            print("开运算的阈值为"+str(threshold_m))
            x = threshold_m
            y = int(threshold_m*1.3)
            ##config##
            Matrix = np.ones((x, y), np.uint8)
            img_edge1 = cv2.morphologyEx(close_img, cv2.MORPH_CLOSE, Matrix)
            img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, Matrix)

            

            return img_edge2 , img

            ##生成预处理图像，车牌识别的第一步

    #微调车牌位置
    def accurate_place(self, card_img_hsv, limit1, limit2, color):
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

    #车牌定位主函数
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
        for i in range(len(colors)-1,-1,-1):
            if colors[i] == "no":
                colors.pop(i)
                card_imgs.pop(i)

         
        #show图片
        # for card_img in card_imgs:
        #     if card_img.any():
        #         cv2.imshow("dingwei", card_img)
        #         cv2.waitKey()
        #         cv2.destroyAllWindows()
        
        return colors,card_imgs  #可能存在多个车牌，暂时保留列表结构                                                                          

    #没用这个方法
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
        lower_green = np.array([70, 20, 80])
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

    #找矩形
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
        for i in range(len(contours)-1,-1,-1):#倒序遍历排除不是车牌的矩形
        #for cnt in contours[:]:
            rect = cv2.minAreaRect(contours[i])# 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）

            area_width,area_height = rect[1]
            if area_width < area_height:
                area_width , area_height = area_height , area_width
            wh_ratio = area_width / area_height #长宽比
            print(wh_ratio)
            #要求矩形区域长宽比在2到5.5之间，2到5.5是车牌的长宽比，其余的矩形排除
            if wh_ratio < 2 or wh_ratio > 6:
                contours.pop(i)
            else:
                car_contours.append(rect)
                box = cv2.boxPoints(rect)
                box = np.int64(box)#int0==int64
                #show红色框框
                oldimg_copy = oldimg.copy()
                oldimg_copy = cv2.drawContours(oldimg_copy, [box], 0, (0, 0, 255), 2)#在原图像上画出矩形,TODO:正式识别时记得删除
                cv2.namedWindow("edge4", cv2.WINDOW_NORMAL)
                cv2.imshow("edge4", oldimg_copy)
                cv2.waitKey()
                cv2.destroyAllWindows()

                #print(rect)
        print("可能存在车牌数："+str(len(car_contours)))#有几个矩形
        return car_contours
                
    def img_Transform(self,card_contours,oldimg):
        img_h, img_w = oldimg.shape[:2]#获取图片的长宽

        card_imgs = []
        #矩形区域可能是倾斜的矩阵，需要矫正，以便使用颜色定位
        for rect in card_contours:#rect((中心点坐标)，（宽，长），角度)
            rect_w,rect_h = rect[1][0],rect[1][1]
            ##config##
            narrow = rect_h/40
            ##config##
            angle = rect[2]#获得矩形旋转的角度
            print("矩形区域的角度是{}".format(angle))
            print("矩形区域宽是{},长是{}".format(rect[1][0],rect[1][1]))



            #如果已经是正的则不需要旋转
            return_flag = False
            if angle == -0:
                return_flag = True
            if angle == -90:
                rect_w, rect_h = rect_h, rect_w
                return_flag = True
            if return_flag:
                card_img = oldimg[int(rect[0][1]-rect_h/2):int(rect[0][1]+rect_h/2),
                          int(rect[0][0]-rect_w/2):int(rect[0][0]+rect_w/2)]
                # cv2.imshow("transform",card_img)
                # cv2.waitKey(0)

                card_imgs.append(card_img)
                continue

            else:


                #如果是歪的
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

                if low_point[0] > height_point[0]: #正角度
                    new_right_point = [right_point[0],height_point[1]]
                    pts2 = np.float32([left_point,height_point,new_right_point])
                    pts1 = np.float32([left_point,height_point,right_point])
                    M = cv2.getAffineTransform(pts1,pts2) #INPUT Array 2*3的变换矩阵
                    dst = cv2.warpAffine(oldimg,M,(round(img_w*2),round(img_h*2)))#仿射变换,OUTPUT Array，输出图像
                    img_math.point_limit(new_right_point)
                    img_math.point_limit(left_point)
                    img_math.point_limit(height_point)
                    card_img = dst[int(left_point[1]):int(height_point[1]),int(left_point[0]+narrow):int(new_right_point[0]-narrow)]#摆正图像
                    #show
                    card_imgs.append(card_img)
                    # cv2.imshow("card2",card_img)
                    # cv2.waitKey(0)
                elif low_point[0] < height_point[0]:  #负角度
                    new_left_point = [left_point[0],height_point[1]]
                    pts2 = np.float32([new_left_point,height_point,right_point])  #字符只是高度需要改变
                    pts1 = np.float32([left_point,height_point,right_point])
                    M = cv2.getAffineTransform(pts1,pts2)
                    dst = cv2.warpAffine(oldimg,M,(round(img_w*2),round(img_h*2)))
                    img_math.point_limit(right_point)
                    img_math.point_limit(height_point)
                    img_math.point_limit(new_left_point)
                    card_img = dst[int(right_point[1]):int(height_point[1]),int(new_left_point[0]+narrow):int(right_point[0]-narrow)]
                    #show
                    card_imgs.append(card_img)
                    # cv2.imshow("card2",card_img)
                    # cv2.waitKey(0)

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
                # cv2.imshow("hsv",card_img_hsv)
                # cv2.waitKey(0)

                row_num , col_num = card_img_hsv.shape[:2]#获取长宽
                card_img_count = row_num * col_num
                for i in range(row_num): #O(N^2)
                    for j in range(col_num):
                        H = card_img_hsv.item(i,j,0)
                        S = card_img_hsv.item(i,j,1)
                        V = card_img_hsv.item(i,j,2)
                        if 11 < H <= 34 and  S > 34:
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

                if color =="green":#绿色本来就区域小，不需要再缩小区域
                    continue

                # #根据车牌颜色在定位，缩小非车牌的边界区域#TODO 不一定要再定位
                # xl, xr, yh, yl = self.accurate_place(card_img_hsv, limit1, limit2, color)
                # if yl == yh and xl == xr:
                #     continue
                # need_accurate = False
                # if yl >= yh:
                #     yl = 0
                #     yh = row_num
                #     need_accurate = True
                # if xl >= xr:
                #     xl = 0
                #     xr = col_num
                #     need_accurate = True
                # card_imgs[card_index] = card_img[yl:yh, xl:xr] if color !="green" or yl < (yh - yl) // 4 else card_img[yl - (yh - yl) // 4:yh,xl:xr]
                #
                #
                # if need_accurate: #可能x或y方向未缩小，需要再试一次
                #     card_img = card_imgs[card_index]
                #     card_img_hsv = cv2.cvtColor(card_img,cv2.COLOR_BGR2HSV)
                #     xl, xr, yh, yl = self.accurate_place(card_img_hsv, limit1, limit2, color)
                #     if yl == yh and xl == xr:
                #         continue
                #     if yl >= yh:
                #         yl = 0
                #         yh = row_num
                #     if xl >= xr:
                #         xl = 0
                #         xr = col_num
                # card_imgs[card_index] = card_img[yl:yh, xl:xr] if color != "green" or yl < (yh - yl) // 4 else card_img[yl - (yh - yl) // 4:yh,xl:xr]

        return colors , card_imgs

    #字符识别
    def char_recogize(self,colors,card_imgs):
        #车牌字符识别
        cards_result=[]#多个车牌的识别结果
        colors_result=[]
        rois = []
        roi = None
        divs = [] #分割图片[[]]
        card_color = None
        for i , color in enumerate(colors):   
            if color in ("blue","yellow","green"):
                card_img = card_imgs[i]
                if card_img.size == 0:
                    continue
                gray_img = cv2.cvtColor(card_img,cv2.COLOR_BGR2GRAY)#转成灰度图
                
                #黄、绿车牌字符比背景暗、与蓝车牌刚好相反，所以黄、绿车牌需要反向
                if color == "green" or color == "yellow":
                    gray_img = cv2.bitwise_not(gray_img)
                ret , gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)#OTSU  ,字符显示的第一步
                #（灰度图，阈值，最大值，阈值类型）把阈值设为0，算法会找到最优阈值
                cv2.imshow("erzhihua", gray_img)
                cv2.waitKey()
                cv2.destroyAllWindows()


                # #img123=np.array(gray_img.convert('L'))
                # plt.figure("lena")
                # arr=gray_img.flatten()
                # n, bins, patches = plt.hist(arr, bins=10, normed=1, facecolor='green', alpha=0.75)
                # plt.show()

                #查找垂直直方图波峰
                x_histogram = np.sum(gray_img , axis=1)
                x_min = np.min(x_histogram)
                x_average = np.sum(x_histogram) / x_histogram.shape[0]
                x_threshold_para = self.cfg["x_threshold_para"]
                x_threshold = (x_min + x_average)/x_threshold_para#TODO:根据分辨率进行调参使用config
                wave_peaks = img_math.find_waves(x_threshold,x_histogram)
                #
                if len(wave_peaks) == 0:
                    print("peek less 0")
                    continue
                #认为水平方向，最大的波峰为车牌区域
                wave = max(wave_peaks,key=lambda x : x[1]- x[0])
                gray_img = gray_img[wave[0]:wave[1]]

                row_num , col_num = gray_img.shape[:2]
                

                if color =="green":
                    gray_img = gray_img[1:row_num + 1]
                gray_img = gray_img[1:col_num - 1]
                #去掉车牌左右边缘的一个像素，防止白边影响阈值判断#TODO，暂时还没想到更好的办法
                y_histogram = np.sum( gray_img, axis=0)
                y_min = np.min(y_histogram)
                y_average = np.sum(y_histogram) / y_histogram.shape[0]
                y_threshold_para = self.cfg["y_threshold_para"]
                y_threshold = (y_min + y_average) / y_threshold_para#U 和 0 要求阈值偏小 ， 否则U和0会被分成两半
                # TODO:根据分辨率进行调参使用config
                print("阈值为："+str(y_threshold))

                wave_peaks = img_math.find_waves(y_threshold,y_histogram)

                print("存在的波峰数量："+str(len(wave_peaks)))
                #车牌的字符应该大于6（蓝、黄7 、 绿8）
                if(len(wave_peaks)<6):
                    print("初始的波峰个数是",len(wave_peaks))
                    continue



                wave = max(wave_peaks, key = lambda x : x[1] - x[0])
                max_wave_dis = wave[1] - wave[0]




                #判断是否是左侧车牌边缘
                if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis / 3 :
                    wave_peaks.pop(0)
                    #wave_peaks[0][0] = 4
                
                #组合分离汉字
                cur_dis = 0
                for i , wave in enumerate(wave_peaks):
                    ##config
                    if wave[1] - wave[0] + cur_dis > int(max_wave_dis * 0.38):#TODO:优化调参
                        break
                    ##config
                    else:
                        cur_dis += wave[1] - wave[0]
                if i > 0:
                    wave = (wave_peaks[0][0],wave_peaks[i][1])
                    wave_peaks = wave_peaks[i + 1:]
                    wave_peaks.insert(0, wave)

                # 去除车牌上的分隔点
                point = wave_peaks[2]  # 第三个点
                if point[1] - point[0] < max_wave_dis / 3:
                    point_img = gray_img[:, point[0]:point[1]]
                    if np.mean(point_img) < 255 / 5:
                        wave_peaks.pop(2)
                
                if len(wave_peaks) < 6:
                    print("分离之后，波峰个数是",len(wave_peaks))
                    continue
                part_cards = img_math.sperate_card(gray_img,wave_peaks) 
                card_color = color
                roi = card_img

                t = Train_SVM.TrainSVM()
                t.train_svm()
                predict_result,div= t.final_rec(part_cards,color)
                divs.append(div)
                colors_result.append(card_color)
                rois.append(roi)
                cards_result.append(predict_result)

        return cards_result, rois, colors_result,divs  # 识别到的字符、定位的车牌图像、车牌颜色

            

if __name__ == '__main__':

    start = time.time()

    q = Predict()
    #if q.isdark("test\\timg.jpg"):
        #print("是黑夜拍的")1
    afterprocess,old=q.preprocess("test\\29.jpg")
    #afterprocess,old =q.preprocess("D:\\车牌测试用\\车牌识别测试图\\P90427-144853.jpg")
    cv2.namedWindow("yuchuli",cv2.WINDOW_NORMAL)
    cv2.imshow("yuchuli", afterprocess)
    cv2.waitKey()
    colors,card_imgs=q.locate_carPlate(afterprocess,old)
    #colors,card_imgs = q.img_only_color(old,afterprocess)
    result , roi , color,divs=q.char_recogize(colors,card_imgs) #all list
    if len(result)==0:
        print("未能识别到车牌")
    else:
        for r in range(len(result)):
            print("车牌的颜色为："+color[r])
            print(result[r])
            print("\n")
    end = time.time()
    print(round(end-start,3))
