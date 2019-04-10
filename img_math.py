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