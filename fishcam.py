# -*- coding: UTF-8 -*-  
import numpy as np
import time
import math

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

threshold_1 = 250
threshold_2 = 200
threshold_3 = 100
point_color = (0,0,0)
thickness = 3
linetype = 8

box = 80

text = ''
fontFace = cv2.FONT_HERSHEY_COMPLEX
fontScale = 1
fontcolor = (0, 255, 0) # BGR
thickness = 1 
lineType = 4
bottomLeftOrigin = 0

# img：要画的圆所在的矩形或图像
# text：要绘制的文字
# org：文字在图像中的左下角坐标
# fontFace：字体，可选 ：FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_PLAIN, FONT_HERSHEY_DUPLEX,FONT_HERSHEY_COMPLEX, FONT_HERSHEY_TRIPLEX, FONT_HERSHEY_COMPLEX_SMALL, FONT_HERSHEY_SCRIPT_SIMPLEX, orFONT_HERSHEY_SCRIPT_COMPLEX, 以上所有类型都可以配合 FONT_HERSHEY_ITALIC使用，产生斜体效果
# fontScale：字体大小，该值和基础大小相乘得到字体大小
# color：文字颜色，如 (0, 0, 255) 红色，BGR
# thickness：字体线条宽度
# lineType：
# 8 (or omitted) ： 8-connected line
# 4：4-connected line
# CV_AA - antialiased line
# bottomLeftOrigin：为 true，图像数据原点在左下角；否则，图像数据原点在左上角


def pre_deal(original_img):
	B = original_img[:,:,0]
	G = original_img[:,:,1]
	R = original_img[:,:,2]

	index = np.array(np.argwhere(R > threshold_1))

	target = []
	for i in index:
		if (G[i[0],i[1]] < threshold_3) and (B[i[0],i[1]] < threshold_2):
			target.append(i)
	target = np.array(target)
	final_center = []
	
	if target.size != 0:
		range_value = [min(target[:,0]),max(target[:,0]),min(target[:,1]),max(target[:,1])]
		ptLeftTop = (range_value[2],range_value[0])
		ptRightBottom = (range_value[3],range_value[1])
		print('box:',range_value[3]-range_value[2], ' ' , range_value[1]-range_value[0])
		if (range_value[3]-range_value[2]) < box and (range_value[1]-range_value[0]) < box :
			final_center = (math.ceil(((range_value[0] + range_value[1])/2)),math.ceil(((range_value[2] + range_value[3])/2)))
		else:
			target_range = original_img[range_value[0]:range_value[1],range_value[2]:range_value[3],:]
			index_target1 = np.array(np.argwhere(target_range[:,:,0] > threshold_1))
			index_target = []
			for i in index_target1:
				if target_range[i[0],i[1],1] > threshold_1 and target_range[i[0],i[1],2] > threshold_1:
					index_target.append(i)
			index_target = np.array(index_target)

			if index_target.size != 0:
				index_target[:,0] = index_target[:,0] + min(target[:,0])
				index_target[:,1] = index_target[:,1] + min(target[:,1])
				final_center = [math.ceil(np.mean(index_target[:,0])),math.ceil(np.mean(index_target[:,1]))]
	final_center = np.array(final_center)
	
	if final_center.size != 0:
		ptLeftTop = (range_value[2],range_value[0])
		ptRightBottom = (range_value[3],range_value[1])
		text = str(range_value[3]-range_value[2]) + ' ' + str(range_value[1]-range_value[0])
		cv2.putText(original_img, text, ptRightBottom, fontFace, fontScale, fontcolor, thickness, lineType)
		cv2.rectangle(original_img, ptLeftTop, ptRightBottom, point_color, thickness, linetype)
		cv2.circle(original_img,(final_center[1],final_center[0]),5,[0,0,0],-1)
	cv2.imshow('RectifiedImage',original_img)

def get_center(original_img):
	text = ""
	B = original_img[:,:,0]
	G = original_img[:,:,1]
	R = original_img[:,:,2]

	index = np.array(np.argwhere(R > threshold_1))

	target = []
	for i in index:
		if (G[i[0],i[1]] < threshold_3) and (B[i[0],i[1]] < threshold_2):
			target.append(i)
	target = np.array(target)

	final_center = []


	if target.size != 0:

		range_value = [min(target[:,0]),max(target[:,0]),min(target[:,1]),max(target[:,1])]
		target_range = original_img[range_value[0]:range_value[1],range_value[2]:range_value[3],:]
		index_target1 = np.array(np.argwhere(target_range[:,:,0] > threshold_1))

		index_target = []
		for i in index_target1:
			if target_range[i[0],i[1],1] > threshold_1 and target_range[i[0],i[1],2] > threshold_1:
				index_target.append(i)
		index_target = np.array(index_target)

		if index_target.size != 0:
			index_target[:,0] = index_target[:,0] + min(target[:,0])
			index_target[:,1] = index_target[:,1] + min(target[:,1])

			final_center = [math.ceil(np.mean(index_target[:,0])),math.ceil(np.mean(index_target[:,1]))]

	final_center = np.array(final_center)
	if final_center.size != 0:
		ptLeftTop = (range_value[2],range_value[0])
		ptRightBottom = (range_value[3],range_value[1])
		text = str(range_value[3]-range_value[2]) + ' ' + str(range_value[1]-range_value[0])
		cv2.putText(original_img, text, ptRightBottom, fontFace, fontScale, fontcolor, thickness, lineType)
		cv2.rectangle(original_img, ptLeftTop, ptRightBottom, point_color, thickness, linetype)
		cv2.circle(original_img,(final_center[1],final_center[0]),5,[255,0,0],-1)
	cv2.imshow('RectifiedImage',original_img)


def get_center_2(original_img):
	text = " "
	B = original_img[:,:,0]
	G = original_img[:,:,1]
	R = original_img[:,:,2]

	index = np.array(np.argwhere(R > threshold_1))

	target = []
	for i in index:
		if (G[i[0],i[1]] < threshold_3) and (B[i[0],i[1]] < threshold_2):
			target.append(i)
	target = np.array(target)

	final_center = []
	if target.size != 0:

		range_value = [min(target[:,0]),max(target[:,0]),min(target[:,1]),max(target[:,1])]
		final_center = (math.ceil(((range_value[0] + range_value[1])/2)),math.ceil(((range_value[2] + range_value[3])/2)))

	final_center = np.array(final_center)
	

	if final_center.size != 0:
		ptLeftTop = (range_value[2],range_value[0])
		ptRightBottom = (range_value[3],range_value[1])
		text = str(range_value[3]-range_value[2]) + ' ' + str(range_value[1]-range_value[0])
		cv2.putText(original_img, text, ptRightBottom, fontFace, fontScale, fontcolor, thickness, lineType)
		cv2.rectangle(original_img, ptLeftTop, ptRightBottom, point_color, thickness, linetype)
		cv2.circle(original_img,(final_center[1],final_center[0]),5,[0,0,0],-1)
	cv2.imshow('RectifiedImage',original_img)
		


if __name__ == '__main__':

	cam = 0
	count = 4  
	# cap=cv2.VideoCapture(cam,cv2.CAP_DSHOW) 
	#cv2.CAP_DSHOW 参数初始化摄像头,否则无法使用更高分辨率,不知道为什么４.2.0报错，
	cap=cv2.VideoCapture(cam)#电脑自带摄像头为0,usb外接摄像头为1

	K = np.loadtxt("/home/dzf/FishCamera/test_parameters_k_d/K.txt")
	D = np.loadtxt("/home/dzf/FishCamera/test_parameters_k_d/D.txt")
	# print('parameter K:\n',K)
	
	# print('parameter K:\n',D)
	
	#获取分辨率
	width,height= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	
	# 设置编码格式
	# cap.set(6,cv2.VideoWriter.fourcc('M','J','P','G'))
	# # 设置摄像头设备帧率
	# cap.set(cv2.CAP_PROP_FPS, 120)
	# cap.set(5,120)
	#优化内参教和畸变系数
	P = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K,D,(width,height),None)

	#此处计算花费时间较大，需从循环中抽取出来
	mapx2,mapy2=cv2.fisheye.initUndistortRectifyMap(K,D,None,P,(width,height),cv2.CV_32F)

	# num_frames=120
	# start = time.time()
	# for i in range(0,num_frames):
	# 	ret,frame_0=cap.read()
	# end = time.time()
	# seconds=end-start
	# print("fps=",num_frames/seconds)

	while (True):
		_,frame = cap.read()
		# cv2.imshow('raw',frame)

		#畸变矫正
		frame_rectified = cv2.remap(frame,mapx2,mapy2,interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)

		cv2.namedWindow("RectifiedImage",cv2.WINDOW_FREERATIO)
		cv2.resizeWindow("RectifiedImage",1280,960)#调整显示窗口大小

		pre_deal(frame_rectified)

		# print('layer_center size:',layer_center.size)

		# cv2.moveWindow('RectifiedImage',800,100)#移动窗口位置

		

		if cv2.waitKey(1) & 0xFF == ord('q'): #1表示延时１ms切换到下一帧　０表示显示当前帧图像
			break 
		if cv2.waitKey(1) & 0xFF == ord('s'):
			cv2.imwrite("/home/dzf/FishCamera/target_img/original_img_" + str(count) + ".jpg", frame)
			cv2.imwrite("/home/dzf/FishCamera/target_img/rectified_img_" + str(count) + ".jpg", frame_rectified)			
			count = count + 1

	cap.release()
	cv2.destroyAllWindows()