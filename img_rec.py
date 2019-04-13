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







