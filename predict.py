import cv2
import numpy as np
from numpy.linalg import norm
import sys
import os
import json
import img_math
import img_rec

SZ = 20 #训练图片长宽 (训练样本是20*20的图片)
MAX_WIDTH = 2000 #原始图片最大宽度
MIN_AREA = 2000 #车牌区域的允许最大面积（不能过大）
PROVINCE_START = 1000

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
    
    #训练svm
    def train(self,samples,responses):
        self.model.trian(samples,cv2.ml.ROW_SAMPLE, responses)

    #字符识别
    def predict(self,samples):
        r = self.model.predict(samples)
        return r[1].ravel()
    


class CardPredictor:
    def __init__(self):
        #可以读取部分参数,便于根据图片的分辨率进行调整
        pass

    def __del__(self):
        self.save_traindata()
    

    def train_svm(self):
        #识别英文字母和数字
        self.model = SVM(C=1,gamma=0.5)#TODO:优化调参，C越大越严格越容易过拟合，gamma过大会导致只支持样本

        #识别中文
        self.modelchinese = SVM(C=1,gamma=0.8)#TODO:优化调参

        #对于字母和数字的训练
        if os.path.exists("svm.dat"):
            self.model.load("svm.dat")
        else:
            chars_train = []#储存训练后的图像
            chars_label = []
            for root,dirs,files in os.walk("train\\chars"):#游走与改文件下的所有内容
                print("正在训练数字、字母字符...")
                if len(os.path.basename(root))>1:
                    continue
                root_int = ord(os.path.basename(root))#返回对应的十进制证书
                for filename in files:
                    filepath = os.path.join(root,filename)
                    digit_img = cv2.imread(filepath)
                    digit_img = cv2.cvtColor(digit_img,cv2.COLOR_BGR2GRAY)
                    chars_train.append(digit_img)
                    chars_label.append(root_int)
            
            chars_train = list(map(img_rec.deskew,chars_train))#对于每个样本，使用图像的二阶矩模型来抗色偏
            chars_train = img_rec.preprocess_hog(chars_train)
            char_label = np.array(chars_label)
            print(chars_train.shape)
            self.model.train(chars_train,chars_label)
        
                

