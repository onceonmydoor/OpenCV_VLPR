# OpenCV_VLPR

2019/5/9日进度报告



## 完成了没优化的基础版本

### **功能如下**

- 单个图片的识别

![1557215109607](https://raw.githubusercontent.com/onceonmydoor/OpenCV_VLPR/master/readme_img/1557215109607.png)

- 数据库的查询（图片识别也会放入数据库）数据库使用的是mysql

  ![1557215152576](https://raw.githubusercontent.com/onceonmydoor/OpenCV_VLPR/master/readme_img/1557215152576.png)

- 车牌号码的查询，颜色+时间区间的查询，条件查询中时间是必选的，优先级比颜色高

- 查询所有的车牌（默认是所有的车牌）

  ![1557215221227](https://github.com/onceonmydoor/OpenCV_VLPR/readme_img/1557215221227.png)

  ![1557215244900](https://github.com/onceonmydoor/OpenCV_VLPR/readme_img/1557215244900.png)

- 显示预处理的图片

  ![1557217000162](https://github.com/onceonmydoor/OpenCV_VLPR/readme_img/1557217000162.png)



- 摄像头模式（每两秒识别一次车牌，可以保存录像，调整各种参数）

  ![1557220554805](.\readme_img\1557220554805.png)



## 所使用到的第三方库：

版本就不列了太多了

opencv,time,qiamge2ndarray,datetime,threading,numpy,json,PIL,pymysql

界面：pyqt5



### 问题记录

- [ ] J和U识别率低，SVM需要调参训练
- [ ] 多车牌问题，应该确认为1 ，即取消多个车牌的识别
- [ ] 美化QSS界面
- [ ] 定位之后左右两边的干扰
- [ ] 绿色车牌识别率低





### 



### 下星期的工作

- 测试，写表格
- 对测试结果进行优化
- 开始初步写论文+画图
- 解决上述问题





## 以下是需要在论文中画出来的流程图

### 预处理步骤

### 车牌粗定位步骤

### 车牌细定位步骤

### 车牌字符分割步骤

### 车牌识别步骤



