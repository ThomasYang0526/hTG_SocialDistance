#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 14:12:01 2021

@author: thomas_yang
"""

import cv2
import numpy as np
import random
import pickle
from scipy.spatial import distance as dist

'''
video config
'''
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('social_distance.avi', fourcc, 30.0, (1000, 480))

# perspective points
person_num = 50
MIN_DISTANCE = 70
color_map = [list(np.random.random(size=3) * 256) for i in range(person_num)]
color = (0, 255, 0)
thickness = 2
isClosed = True
with open('stereo_calibration.pickle', 'rb') as f:
      cal_data = pickle.load(f)

'''
point for target square in calibration image
'''
# pts_src = np.array([[304, 264], [382, 334], 
#                     [171, 378], [120, 284],
#                     ], np.int32)

pts_src = []
cropping = False
def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global pts_src, cropping
	if event == cv2.EVENT_LBUTTONDOWN:
		pts_src.append((x, y))
		cv2.circle(view_ori, (x, y), 5, (0, 255, 0), -1)
		cv2.imshow("social_distance", view_ori)

view_ori = cv2.imread('./images/img_corr.jpg')
clone = view_ori.copy()
cv2.namedWindow("social_distance")
cv2.setMouseCallback("social_distance", click_and_crop)
while True:
	cv2.imshow("social_distance", view_ori)
	key = cv2.waitKey(1) & 0xFF
	if len(pts_src) == 4 or key == ord("c"):
		pts_src = np.array(pts_src).astype(np.int32)
		view_ori = cv2.polylines(view_ori, [pts_src], isClosed, color, thickness)
		cv2.imshow("social_distance", view_ori)
		cv2.waitKey(10)
		break

# cv2.destroyAllWindows()   
     

'''
social distance part
'''        
# project pts_dst
shift_x, shift_y = 200, 400
pts_dst = np.array([[0, 0], [70, 0], 
                    [70, 100], [0, 100],
                    ], np.int32) + [shift_x, shift_y]

view_ori = cv2.polylines(view_ori, [pts_src], isClosed, color, thickness)
      
view_land_ori = view_ori*0
h_l,  w_l = view_land_ori.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cal_data["M1"], cal_data["dist1"], (w_l,h_l), 1, (w_l,h_l))
view_land_ori = cv2.rectangle(view_land_ori, (0, 0) , (640, 480), (0, 255, 255), thickness)
view_land_ori = cv2.undistort(view_land_ori, cal_data["M1"], cal_data["dist1"], None, newcameramtx) 
view_land_ori = cv2.polylines(view_land_ori, [pts_src], isClosed, color, thickness)

M = cv2.getPerspectiveTransform(pts_src.astype(np.float32), pts_dst.astype(np.float32));
view_land_warp = cv2.warpPerspective(view_land_ori, M, (480, 640))

# random direction 
dirt = np.sin(np.pi*np.random.random(size=person_num)*2) * 1 #np.random.random(size=person_num)
dirt = np.expand_dims(np.tile(np.expand_dims(dirt, 1), 2), 1)
dirt *= (np.random.randint(low=-2, high=3, size=dirt.shape) + np.random.random(size=dirt.shape))


# random point simulate person detected
persons_ori = []
for i in range(person_num):
    persons_ori.append([random.randint(50, 500), random.randint(50, 400)])

persons_ori = np.float32(persons_ori).reshape(-1, 1, 2)
for it in range(1000):
    print(it)
    view_ori_c = view_ori.copy()
    view_warp_c = view_land_warp.copy()
    persons_warp = cv2.perspectiveTransform(persons_ori, M).astype(np.int32)
    D = dist.cdist(persons_warp.squeeze(), persons_warp.squeeze(), metric="euclidean")
    for i in range(0, D.shape[0]):
        for j in range(i + 1, D.shape[1]):
            if D[i, j] < MIN_DISTANCE:
                center_i = persons_warp[i]
                center_j = persons_warp[j]
                cv2.line(view_warp_c, 
                          (center_i[0][0], center_i[0][1]),
                          (center_j[0][0], center_j[0][1]),
                          (0, 0, 255), 2)
                
    for idx, (center_ori, center_warp) in enumerate(zip(persons_ori, persons_warp)):
        center_ori = center_ori.astype(np.int32).squeeze()
        center_warp = center_warp.squeeze()
        cv2.circle(view_ori_c, center_ori, 5, color_map[idx], -1)
        cv2.line(view_ori_c, center_ori, center_ori+(0, -20), color_map[idx],thickness)
        cv2.circle(view_warp_c, center_warp, 5, color_map[idx], -1)
    # shift = np.random.randint(low=-1, high=2, size=(persons_ori.shape[0], 1, 2))
    persons_ori += dirt

    view_warp_c = cv2.resize(view_warp_c, (360, 480))
    social_distance = cv2.hconcat((view_ori_c, view_warp_c))
    # cv2.imshow('view_ori_c', view_ori_c)
    # cv2.imshow('view_warp_c', view_warp_c)
    cv2.imshow('social_distance', social_distance)
    out.write(social_distance)

    # 按下 q 鍵離開迴圈
    if cv2.waitKey(10) == ord('q'):
        break
out.release()
cv2.destroyAllWindows()