import cv2
import numpy as np
from numpy.linalg import norm
import sys
import os
import json

#支持向量机
SZ = 20  #训练图片长宽
MAX_WIDTH = 1000 #原始图片的最大宽度
MIN_AREA = 2000 #车牌区域的允许最大面积
PROVINCE_START = 1000

# 来自opencv的sample，用于svm训练-----校正图像
def deskew(img):#在找HOG之前，使用图像的二阶矩模型来抗色偏。首先定义一个函数deskew()取一个数字图像并对他抗色偏
    m = cv2.moments(img)#获取图像的图像矩 即为图像的各类几何特征
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

# 来自opencv的sample，用于svm训练
def preprocess_hog(digits):#方向梯度的直方图(HOG）作为特征向量
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)#每个单元在X和Y方向的Sobel导数
        mag, ang = cv2.cartToPolar(gx, gy)# 笛卡尔坐标转换为极坐标, → magnitude, angle
        bin_n = 16
        # quantizing binvalues in (0...16)
        bin = np.int32(bin_n * ang / (2 * np.pi))
         # Divide to 4 sub-squares
        bin_cells = bin[:10, :10], bin[10:, :10], bin[:10, 10:], bin[10:, 10:]
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)


provinces = [
    "zh_cuan", "川",
    "zh_e", "鄂",
    "zh_gan", "赣",
    "zh_gan1", "甘",
    "zh_gui", "贵",
    "zh_gui1", "桂",
    "zh_hei", "黑",
    "zh_hu", "沪",
    "zh_ji", "冀",
    "zh_jin", "津",
    "zh_jing", "京",
    "zh_jl", "吉",
    "zh_liao", "辽",
    "zh_lu", "鲁",
    "zh_meng", "蒙",
    "zh_min", "闽",
    "zh_ning", "宁",
    "zh_qing", "青",
    "zh_qiong", "琼",
    "zh_shan", "陕",
    "zh_su", "苏",
    "zh_sx", "晋",
    "zh_wan", "皖",
    "zh_xiang", "湘",
    "zh_xin", "新",
    "zh_yu", "豫",
    "zh_yu1", "渝",
    "zh_yue", "粤",
    "zh_yun", "云",
    "zh_zang", "藏",
    "zh_zhe", "浙"
]



