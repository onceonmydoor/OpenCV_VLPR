"""
用于SVM的模型训练
"""
import cv2
import numpy as np
from numpy.linalg import norm
import sys
import os
import json
import img_math


SZ = 20 #训练图片长宽 (训练样本是20*20的图片)
MAX_WIDTH = 2000 #原始图片最大宽度
MIN_AREA = 2000 #车牌区域的允许最大面积（不能过大）
PROVINCE_START = 1000 #用于字符分类，把中文和数字字母错开

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



class StartModel(object):
    def load(self,fn):
        self.model = self.model.load(fn)

    def save(self,fn):
        self.model.save(fn)

class SVM(StartModel):
    def __init__(self, C=1 , gamma =0.5):#svm C核函数和gamma参数默认
        self.model = cv2.ml.SVM_create()#初始化svm模型
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)
    
    #训练svm，样本+标签 
    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    #字符识别
    def predict(self,samples):
        r = self.model.predict(samples)
        return r[1].ravel()
    
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

class TrainSVM:
    
    def __del__(self):
        self.save_traindata()
    

    def train_svm(self):
        #识别英文字母和数字
        self.model = SVM(C=1,gamma=0.5)#TODO:优化调参，C越大越严格越容易过拟合，gamma过大会导致只支持样本

        #识别中文
        self.modelchinese = SVM(C=1,gamma=0.6)#TODO:优化调参

        #对于字母和数字的训练
        if os.path.exists("svm\\svm.dat"):
            self.model.load("svm\\svm.dat")
        else:
            chars_train = []#储存训练后的图像矩阵
            chars_label = []#该数字的标签
            for root,dirs,files in os.walk("train\\chars2"):#游走与改文件下的所有内容
                print("正在训练数字、字母字符...")
                if len(os.path.basename(root))>1:
                    continue
                root_int = ord(os.path.basename(root))#返回对应的Ascii码
                for filename in files:
                    filepath = os.path.join(root,filename)
                    digit_img = cv2.imread(filepath)#读取训练图像
                    digit_img = cv2.cvtColor(digit_img,cv2.COLOR_BGR2GRAY)#转换成灰度图
                    chars_train.append(digit_img)
                    chars_label.append(root_int)
            
            chars_train = list(map(deskew,chars_train))#对于每个样本，使用图像的二阶矩模型来抗色偏
            chars_train = preprocess_hog(chars_train)
            chars_label = np.array(chars_label)
            print(chars_train.shape)
            self.model.train(chars_train,chars_label)


        #对于中文字符的训练
        if os.path.exists("svm\\svmChinese.dat"):
            self.modelchinese.load("svm\\svmChinese.dat")
        else:
            charsC_train = [] 
            charsC_label = []#该provinces的索引
            for root, dirs, files in os.walk("train\\charsChinese"):
                print("正在识别中文字符...")
                if not os.path.basename(root).startswith("zh_"):
                    continue
                pinyin = os.path.basename(root)
                index = provinces.index(pinyin) + PROVINCE_START + 1  # +1是拼音对应的汉字
                for filename in files:
                    filepath = os.path.join(root, filename)
                    digitC_img = cv2.imread(filepath)
                    digitC_img = cv2.cvtColor(digitC_img, cv2.COLOR_BGR2GRAY)
                    charsC_train.append(digitC_img)
                    # chars_label.append(1)
                    charsC_label.append(index)
            charsC_train = list(map(deskew,charsC_train))
            charsC_train = preprocess_hog(charsC_train)
            charsC_label = np.array(charsC_label)
            print(charsC_train.shape)
            self.modelchinese.train(charsC_train,charsC_label)
        
    
    def save_traindata(self):
        if not os.path.exists("svm\\svm.dat"):
            self.model.save("svm\\svm.dat")
        if not os.path.exists("svm\\svmChinese.dat"):
            self.modelchinese.save("svm\\svmChinese.dat")



    def final_rec(self,part_cards,color):
        predict_result = []
        div = []
        
        for i , part_card in enumerate(part_cards):

            #排除固定车牌的铆钉
            if np.mean(part_card) < 255 /5:
                print("a point")
                continue
            part_card_old = part_card
            w = abs(part_card.shape[1]-SZ) // 2
            #
            part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0,0,0])
            part_card = cv2.resize(part_card,(SZ,SZ),interpolation=cv2.INTER_AREA)
            #卷积边界，填充边界使分割图像变成训练集大小，即20*20


            #显示每个分割字符，用于界面显示
            # div.append(part_card)
            # cv2.imshow("fengezifu",part_card)
            # cv2.waitKey(0)

            part_card = preprocess_hog([part_card])  
            if i == 0 :
                #识别第一中文字符
                resp = self.modelchinese.predict(part_card)
                charactor = provinces[int(resp[0])-PROVINCE_START]
            else:
                resp = self.model.predict(part_card)
                charactor = chr(resp[0])
            #判断最后一个数是否是车牌的边缘，假设车牌的边缘被认为是1
            if i == len(part_cards) - 1 and (charactor == "1" or charactor=="Z" or charactor == "7"):
                if color != "green" and len(predict_result)==7:
                    #只有绿色车牌是8位数
                    continue
                if color == "green" and len(predict_result)==8:
                #绿色车牌已经是8位数了
                    continue
                # if part_card_old.shape[0] / part_card_old.shape[1] >= 7 : #如果1太细，认为是边缘
                #     continue
            predict_result.append(charactor)
        return predict_result,div # 识别到的字符、定位的车牌图像、车牌颜色

# if __name__ == '__main__':
#     c = TrainSVM()
#     c.train_svm()  
#     del c
                

