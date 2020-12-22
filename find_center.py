import math

import numpy as np

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2 

point_color = (0,0,0)
thickness = 3
linetype = 8


def get_center():
    threshold_1 = 252
    threshold_2 = 200
    threshold_3 = 130

    # original_img = cv2.imread('/home/dzf/FishCamera/target_img/original_img_1.jpg')
    original_img = cv2.imread('/home/dzf/FishCamera/target_img/rectified _img_2.jpg')
    cv2.imshow('original_img',original_img)

    B = original_img[:,:,0]
    G = original_img[:,:,1]
    R = original_img[:,:,2]
    # python opencv2 RGB

    # print(original_img.shape)

    index = np.array(np.argwhere(R > threshold_1))

    target =[]
    for i in index:
        if (G[i[0],i[1]] < threshold_3) and (B[i[0],i[1]] < threshold_2):
            target.append(i)
        # if G[i[0],i[1]] < threshold_3:
        #     print(G[i[0],i[1]])
    target = np.array(target)

    final_center = []

    if target.size != 0:
        range_value = [min(target[:,0]),max(target[:,0]),min(target[:,1]),max(target[:,1])]
        target_range = original_img[range_value[0]:range_value[1],range_value[2]:range_value[3],:]
        
        # print(target_range)
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
            cv2.circle(original_img,(final_center[1],final_center[0]),5,[0,0,0],-1)
            #画矩形
            ptLeftTop = (range_value[2],range_value[0])
            ptRightBottom = (range_value[3],range_value[1])

            center2 = (math.ceil(((range_value[2] + range_value[3])/2)),math.ceil(((range_value[0] + range_value[1])/2)))

            cv2.circle(original_img,(center2[0],center2[1]),5,[0,0,0],-1)
            cv2.rectangle(original_img, ptLeftTop, ptRightBottom, point_color, thickness, linetype)
    # print(final_center)

    
    cv2.imshow('original_img',original_img)
    cv2.waitKey(0)

def open_jpg():
    original_img = cv2.imread('/home/dzf/FishCamera/target_img/rectified_img_4.jpg')
    cv2.imshow('original_img',original_img)
    cv2.waitKey(0)

if __name__ == '__main__':

    get_center()
    # open_jpg()