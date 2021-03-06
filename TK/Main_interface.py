import tkinter as tk
from tkinter.filedialog import *
from tkinter import ttk
import predict
import img_math
import cv2
from PIL import Image, ImageTk
import threading
import time
import Train_SVM



class Surface(ttk.Frame):
	pic_path = ""
	viewhigh = 600
	viewwide = 600
	update_time = 0
	thread = None
	thread_run = False
	camera = None
	color_transform = {"green":("绿牌","#55FF55"), "yellow":("黄牌","#FFFF00"), "blue":("蓝牌","#6666FF")}
		
	def __init__(self, win):
		ttk.Frame.__init__(self, win)
		frame_left = ttk.Frame(self)
		frame_right1 = ttk.Frame(self)
		frame_right2 = ttk.Frame(self)
		win.title("车牌识别")
		win.state("zoomed")
		self.pack(fill=tk.BOTH, expand=tk.YES, padx="5", pady="5")
		frame_left.pack(side=LEFT,expand=1,fill=BOTH)#grid分区，左部分为图片
		frame_right1.pack(side=TOP,expand=1,fill=tk.Y)#grid分区，右上部分为显示区域
		frame_right2.pack(side=RIGHT,expand=0)#grid分区，右下部分为按钮部分
		ttk.Label(frame_left, text='原图：').pack(anchor="nw") 
		ttk.Label(frame_right1, text='车牌位置：').grid(column=0, row=0, sticky=tk.W)
		
		from_pic_ctl = ttk.Button(frame_right2, text="来自图片", width=20, command=self.from_pic)
		from_vedio_ctl = ttk.Button(frame_right2, text="来自摄像头", width=20, command=self.from_vedio)
		#from_img_pre = ttk.Label(frame_right2,text="查看形状预处理图像",width=20,command=self.show_img_pre)
		self.image_ctl = ttk.Label(frame_left)
		self.image_ctl.pack(anchor="nw")
		
		self.roi_ctl = ttk.Label(frame_right1)
		self.roi_ctl.grid(column=0, row=1, sticky=tk.W)
		ttk.Label(frame_right1, text='识别结果：').grid(column=0, row=2, sticky=tk.W)
		self.result_ctl = ttk.Label(frame_right1, text="")
		self.result_ctl.grid(column=0, row=3, sticky=tk.W)
		self.color_ctl = ttk.Label(frame_right1, text="", width="20")
		self.color_ctl.grid(column=0, row=4, sticky=tk.W)
		from_vedio_ctl.pack(anchor="se", pady="5")
		from_pic_ctl.pack(anchor="se", pady="5")
		self.predictor = predict.Predict()
		#self.Train_SVM.train_svm()
		
	def get_imgtk(self, img_bgr):#显示出需要识别的图像
		img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
		im = Image.fromarray(img)
		imgtk = ImageTk.PhotoImage(image=im)
		wide = imgtk.width()
		high = imgtk.height()
		if wide > self.viewwide or high > self.viewhigh:
			wide_factor = self.viewwide / wide
			high_factor = self.viewhigh / high
			factor = min(wide_factor, high_factor)
			wide = int(wide * factor)
			if wide <= 0 : wide = 1
			high = int(high * factor)
			if high <= 0 : high = 1
			im=im.resize((wide, high), Image.ANTIALIAS)
			imgtk = ImageTk.PhotoImage(image=im)
		return imgtk
	
	def show_roi(self, result, roi, color):#显示出车牌的定位图像，显示识别的结果
		"""
		@parameter:
		result : 识别字符
		roi : 定位的车牌图像
		color : 车牌颜色
		"""
		if result :
			roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
			roi = Image.fromarray(roi)
			self.imgtk_roi = ImageTk.PhotoImage(image=roi)
			self.roi_ctl.configure(image=self.imgtk_roi, state='enable')
			self.result_ctl.configure(text=str(result))
			self.update_time = time.time()
			try:
				c = self.color_transform[color]
				self.color_ctl.configure(text=c[0], background=c[1], state='enable')
			except: 
				self.color_ctl.configure(state='disabled')
		elif self.update_time + 8 < time.time():
			self.roi_ctl.configure(state='disabled')
			self.result_ctl.configure(text="未能成功识别")
			self.color_ctl.configure(state='disabled')
	
	#开启摄像头
	def from_vedio(self):
		if self.thread_run:
			return
		if self.camera is None:
			self.camera = cv2.VideoCapture(0)
			if not self.camera.isOpened():
				tk.Message('警告！没有摄像头或者未打开摄像头')
				self.camera = None
				return
		self.thread = threading.Thread(target=self.vedio_thread, args=(self,))
		self.thread.setDaemon(True)
		self.thread.start()
		self.thread_run = True
		
	def from_pic(self):#选择图片之后，直接对其识别
		self.thread_run = False
		self.pic_path = askopenfilename(title="选择识别图片", filetypes=[("jpg图片", "*.jpg")])
		if self.pic_path:
			img_bgr = predict.imreadex(self.pic_path)
			self.imgtk = self.get_imgtk(img_bgr)
			self.image_ctl.configure(image=self.imgtk)
			r, roi, color = self.predictor.Predict(img_bgr)
			self.show_roi(r, roi, color)


	@staticmethod
	def vedio_thread(self):#摄像识别的线程
		self.thread_run = True
		predict_time = time.time()
		while self.thread_run:#无限循环，获取图片
			_, img_bgr = self.camera.read()
			self.imgtk = self.get_imgtk(img_bgr)
			self.image_ctl.configure(image=self.imgtk)
			if time.time() - predict_time > 2:#每2秒识别一次
				result, roi, color = self.predictor.predict(img_bgr)
				self.show_roi(result, roi, color)
				predict_time = time.time()
		print("run end")
		
		
def close_window():
	print("destroy")
	if surface.thread_run :
		surface.thread_run = False
		surface.thread.join(2.0)#检验线程是否结束，没有结束就阻塞2秒
	win.destroy()#close，退出输出destroy
	
	
if __name__ == '__main__':
	win=tk.Tk()
	surface = Surface(win)
	win.protocol('WM_DELETE_WINDOW', close_window)
	#消息循环
	win.mainloop()
	
