#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 19:22:38 2021

@author: thomas_yang
"""

import cv2
import pickle
import numpy as np

cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

with open('stereo_calibration.pickle', 'rb') as f:
      cal_data = pickle.load(f)

color = (0, 255, 0)
thickness = 2
isClosed = True
pts = np.array([[304, 264], [382, 334], 
                [171, 378], [120, 284],
                ],np.int32)

while(True):
    # 擷取影像
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    h_l,  w_l = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cal_data["M1"], cal_data["dist1"], (w_l,h_l), 1, (w_l,h_l))
    dst_l = cv2.undistort(frame, cal_data["M1"], cal_data["dist1"], None, newcameramtx)    
    dst_l = cv2.polylines(dst_l, [pts], isClosed, color, thickness)

    # 顯示圖片
    cv2.imshow('live', frame)
    cv2.imshow("Left0", dst_l)
    cv2.imwrite('img_dist.jpg', frame)
    cv2.imwrite('img_corr.jpg', dst_l)

    # 按下 q 鍵離開迴圈
    if cv2.waitKey(1) == ord('q'):
        break

# # 釋放該攝影機裝置
cap.release()
cv2.destroyAllWindows()


      
  