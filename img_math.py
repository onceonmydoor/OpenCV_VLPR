import cv2
import numpy as np


MAX_WIDTH = 1000
MIN_AREA = 2000
SZ = 20
PROVINCE_START = 1000
"""
一些方法的定义
- 文件读取函数
- 取零值函数
- 矩阵矫正函数
- 颜色判断函数
"""

def img_read(filename):
    return cv2.imdecode(np.fromfile(filename,dtype=np.uint8),cv2.IMREAD_COLOR)
    #使用uint8方式读取文件 放入imdecode中,解决imread不能读取中文路径的问题

def find_waves(threshold,histogram):
    up_point = -1 #上升点
    is_peak = False 
    if histogram[0] > threshold:
        up_point = 0 ;
        is_peak = True
    wave_peaks = []
    for i,x in enumerate(histogram):
        if is_peak and x < threshold:
            if i - up_point > 2:
                is_peak = False
                wave_peaks.append((up_point,i))#以元组的形式保存
        elif not is_peak and x >= threshold:
            is_peak = True
            up_point = i
    if is_peak and up_point != -1 and i-up_point > 4:
        wave_peaks.append((up_point,i))
    return wave_peaks

def point_limit(point):
    if point[0] < 0:
        point[0] = 0
    if point[1] < 0:
        point[1] = 0

def accurate_place(card_img_hsv,limit1,limit2,color):#车牌根据颜色再定位
    row_num , col_num = card_img_hsv.shape[:2]
    xl = col_num 
    xr = 0 
    yh = 0
    yl = row_num
    row_num_limit = 20
    col_num_limit = col_num * 0.8 if color != "green" else col_num * 0.5 #绿色车牌存在渐变,判断绿色暂时判断一半
    for i in range(row_num): #O(N^2)
        count = 0
        for j in range(col_num):
            H = card_img_hsv.item(i,j,0)
            S = card_img_hsv.item(i,j,1)
            V = card_img_hsv.item(i,j,2)
            if limit1 < H <= limit2 and 34 < S and 46 < V:
                count += 1 
        if count > col_num_limit :
            if yl > i:
                yl = i
            if yh < i:
                yh = i

    for j in range(col_num):
        count = 0
        for i in range(row_num):
            H = card_img_hsv.item(i,j,0)
            S = card_img_hsv.item(i,j,1)
            V = card_img_hsv.item(i,j,2)
            if limit1 < H <= limit2 and 34 < S and 46 < V:
                count += 1 
        if count > col_num_limit :
            if xl > i:
                xl = i
            if xr < i:
                xr = i
    return xl,xr,yh,yl




def verify_scale(rotate_rect):
    error = 0.4
    aspect = 4  # 4.7272
    min_area = 10 * (10 * aspect)
    max_area = 150 * (150 * aspect)
    min_aspect = aspect * (1 - error)
    max_aspect = aspect * (1 + error)
    theta = 30

    # 宽或高为0，不满足矩形直接返回False
    if rotate_rect[1][0] == 0 or rotate_rect[1][1] == 0:
        return False

    r = rotate_rect[1][0] / rotate_rect[1][1]
    r = max(r, 1 / r)
    area = rotate_rect[1][0] * rotate_rect[1][1]
    if area > min_area and area < max_area and r > min_aspect and r < max_aspect:
        # 矩形的倾斜角度在不超过theta
        if ((rotate_rect[1][0] < rotate_rect[1][1] and rotate_rect[2] >= -90 and rotate_rect[2] < -(90 - theta)) or
                (rotate_rect[1][1] < rotate_rect[1][0] and rotate_rect[2] > -theta and rotate_rect[2] <= 0)):
            return True
    return False



def sperate_card(img,waves):
    part_cards = []
    for wave in waves:
        part_cards.append(img[:,wave[0]:wave[1]])
    return part_cards 


