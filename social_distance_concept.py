#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 14:08:59 2021

@author: thomas_yang
"""

import cv2
import numpy as np
import random
import pickle
from scipy.spatial import distance as dist


class SocialDistance(object):
    def __init__(self, pickle_file, image_path):
        self.pickle_file = pickle_file
        self.image_path = image_path
        self.pts_src = []
        self.person_num = 50
        self.MIN_DISTANCE = 70
        self.color_map = [list(np.random.random(size=3) * 256) for i in range(self.person_num)]
        self.color = (0, 255, 0)
        self.thickness = 2
        self.isClosed = True        
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter('social_distance.avi', self.fourcc, 30.0, (1000, 480)) 

        self.shift_x, self.shift_y = 200, 200
        self.pts_dst = np.array([[0, 0], [70, 0], 
                                 [70, 100], [0, 100],
                                 ], np.int32) + [self.shift_x, self.shift_y]

        with open(pickle_file, 'rb') as f:
              self.cal_data = pickle.load(f)
        
        self.view_ori = cv2.imread(image_path)              
        self.view_land_ori = self.view_ori*0

        # random direction 
        self.dirt = np.sin(np.pi*np.random.random(size=self.person_num)*2) * 1
        self.dirt = np.expand_dims(np.tile(np.expand_dims(self.dirt, 1), 2), 1)
        self.dirt *= (np.random.randint(low=-2, high=3, size=self.dirt.shape) + np.random.random(size=self.dirt.shape))

        # random point simulate person detected
        self.persons_ori = []
        for i in range(self.person_num):
            self.persons_ori.append([random.randint(0, self.view_land_ori.shape[1]), random.randint(0, self.view_land_ori.shape[0])])
        self.persons_ori = np.float32(self.persons_ori).reshape(-1, 1, 2)
        
    def click_and_crop(self, event, x, y, flags, param):
    	# grab references to the global variables    	
    	if event == cv2.EVENT_LBUTTONDOWN:
    		self.pts_src.append((x, y))
    		cv2.circle(self.view_ori, (x, y), 5, (0, 255, 0), -1)
    		cv2.imshow("social_distance", self.view_ori)        

    def manual_draw_bbox(self):        
        h_l,  w_l = self.view_ori.shape[:2]
        self.newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.cal_data["M1"], self.cal_data["dist1"], (w_l,h_l), 1, (w_l,h_l))
        self.view_ori = cv2.undistort(self.view_ori, self.cal_data["M1"], self.cal_data["dist1"], None, self.newcameramtx) 
        cv2.namedWindow("social_distance")
        cv2.setMouseCallback("social_distance", self.click_and_crop)
        while True:
        	cv2.imshow("social_distance", self.view_ori)
        	key = cv2.waitKey(1) & 0xFF
        	if len(self.pts_src) == 4 or key == ord("c"):
        		self.pts_src = np.array(self.pts_src).astype(np.int32)
        		self.view_ori = cv2.polylines(self.view_ori, [self.pts_src], self.isClosed, self.color, self.thickness)
        		cv2.imshow("social_distance", self.view_ori)
        		cv2.waitKey(10)
        		break        

    def simulation_(self):
        self.view_ori = cv2.polylines(self.view_ori, [self.pts_src], self.isClosed, self.color, self.thickness)        
        self.view_land_ori = cv2.rectangle(self.view_land_ori, (0, 0) , (self.view_land_ori.shape[1], self.view_land_ori.shape[0]), (0, 255, 255), self.thickness)
        self.view_land_ori = cv2.undistort(self.view_land_ori, self.cal_data["M1"], self.cal_data["dist1"], None, self.newcameramtx) 
        self.view_land_ori = cv2.polylines(self.view_land_ori, [self.pts_src], self.isClosed, self.color, self.thickness) 
        M = cv2.getPerspectiveTransform(self.pts_src.astype(np.float32), self.pts_dst.astype(np.float32));
        view_land_warp = cv2.warpPerspective(self.view_land_ori, M, (self.view_land_ori.shape[1],self. view_land_ori.shape[0])) 
      
        for it in range(1000):
            print(it)
            self.view_ori_c = self.view_ori.copy()
            self.view_warp_c = view_land_warp.copy()
            persons_warp = cv2.perspectiveTransform(self.persons_ori, M).astype(np.int32)
            D = dist.cdist(persons_warp.squeeze(), persons_warp.squeeze(), metric="euclidean")
            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
                    if D[i, j] < self.MIN_DISTANCE:
                        center_i = persons_warp[i]
                        center_j = persons_warp[j]
                        cv2.line(self.view_warp_c, 
                                 (center_i[0][0], center_i[0][1]),
                                 (center_j[0][0], center_j[0][1]),
                                 (0, 0, 255), 2)
                        
            for idx, (center_ori, center_warp) in enumerate(zip(self.persons_ori, persons_warp)):
                center_ori = center_ori.astype(np.int32).squeeze()
                center_warp = center_warp.squeeze()
                cv2.circle(self.view_ori_c, center_ori, 5, self.color_map[idx], -1)
                cv2.line(self.view_ori_c, center_ori, center_ori+(0, -20), self.color_map[idx], self.thickness)
                cv2.circle(self.view_warp_c, center_warp, 5, self.color_map[idx], -1)
            shift = np.random.randint(low=-1, high=2, size=(self.persons_ori.shape[0], 1, 2))
            self.persons_ori += self.dirt
        
            # view_warp_c = cv2.resize(view_warp_c, (360, 480))
            # social_distance = cv2.hconcat((view_ori_c, view_warp_c))
            cv2.imshow('view_ori_c', self.view_ori_c)
            cv2.imshow('view_warp_c', self.view_warp_c)
            # cv2.imshow('social_distance', social_distance)
            # out.write(social_distance)
        
            # 按下 q 鍵離開迴圈
            if cv2.waitKey(10) == ord('q'):
                break
        self.out.release()
        cv2.destroyAllWindows()
        
if __name__ == '__main__':
    picke_file = 'stereo_calibration.pickle'
    img_path = './images/img_dist.jpg'
    sd = SocialDistance(picke_file, img_path)
    sd.manual_draw_bbox()
    sd.simulation_()
        
        
        
        