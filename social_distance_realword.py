#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 15:21:16 2021

@author: thomas_yang
"""

from social_distance_simulation import SocialDistanceBase
import os
import torch
from config import CLASSES
from torchvision.models import detection
import cv2
import numpy as np
from scipy.spatial import distance as dist
import time

class SocialDistanceReal(SocialDistanceBase):
    def __init__(self, pickle_file, image_path, shift_xy, pts_dst):
        super().__init__(pickle_file, image_path, shift_xy, pts_dst)
        
        self.COLORS = np.random.uniform(100, 255, size=(len(CLASSES), 3))
        self.Confidence_thres = 0.4
        self.MIN_DISTANCE = 40
        self.color_idx = 2
        self.grid_color = (40, 40, 40)
        self.downsample_ratio = 4
        self.model_input_size = (480, 270)

        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.MODELS = {
            'frcnn-resnet': detection.fasterrcnn_resnet50_fpn,
            'frcnn-mobilenet': detection.fasterrcnn_mobilenet_v3_large_320_fpn,
            'retinanet': detection.retinanet_resnet50_fpn
            }

        self.model = self.MODELS["frcnn-mobilenet"](pretrained=True, progress=True, num_classes=91, pretrained_backbone=True).to(self.DEVICE)
        self.model.eval()

    def get_view_homography(self, frame_1st):
        cv2.namedWindow("social_distance")
        cv2.setMouseCallback("social_distance", self.click_and_crop)
        self.view_ori = frame_1st.copy()
        self.view_ori = cv2.undistort(self.view_ori, self.cal_data["M1"], self.cal_data["dist1"], None, self.newcameramtx) 
        while True:
        	cv2.imshow("social_distance", self.view_ori )
        	key = cv2.waitKey(1) & 0xFF
        	if len(self.pts_src) == 4 or key == ord("c"):
        		self.pts_src = np.array(self.pts_src).astype(np.int32)
        		self.view_ori  = cv2.polylines(self.view_ori , [self.pts_src], self.isClosed, self.target_color, self.thickness)
        		cv2.imshow("social_distance", self.view_ori )
        		cv2.waitKey(10)                        
        		break                  
        
        self.view_ori  = cv2.polylines(self.view_ori , [self.pts_src], self.isClosed, self.target_color, self.thickness)
        view_land_ori = self.view_ori *0
        view_land_ori = cv2.rectangle(view_land_ori, (0, 0) , (view_land_ori.shape[1], view_land_ori.shape[0]), (0, 255, 255), self.thickness+1)
        view_land_ori = cv2.undistort(view_land_ori, self.cal_data["M1"], self.cal_data["dist1"], None, self.newcameramtx) 
        self.pts_src //= self.downsample_ratio
        M = cv2.getPerspectiveTransform(self.pts_src.astype(np.float32), self.pts_dst.astype(np.float32));
        view_land_ori = cv2.resize(view_land_ori, self.model_input_size)
        view_land_warp = cv2.warpPerspective(view_land_ori, M, (600, 1200))
        cv2.destroyAllWindows()         
        
        for i in range(self.shift_x, -1, -30):
            cv2.line(view_land_warp, (i, 0), (i, view_land_warp.shape[0]), self.grid_color, 1)
        for i in range(self.shift_x, view_land_warp.shape[1], 30):
            cv2.line(view_land_warp, (i, 0), (i, view_land_warp.shape[0]), self.grid_color, 1)
        for i in range(self.shift_y, -1, -30):
            cv2.line(view_land_warp, (0, i), (view_land_warp.shape[1], i), self.grid_color, 1)
        for i in range(self.shift_y, view_land_warp.shape[0], 30):
            cv2.line(view_land_warp, (0, i), (view_land_warp.shape[1], i), self.grid_color, 1)
        view_land_warp = cv2.rectangle(view_land_warp, (self.shift_x, self.shift_y) , 
                                       (self.shift_x+30, self.shift_y+30), self.target_color, self.thickness)

        return M, view_land_warp  

    def preprocess(self, frame):
        image = frame
        image = cv2.undistort(image, self.cal_data["M1"], self.cal_data["dist1"], None, self.newcameramtx) 
        image = cv2.resize(image, self.model_input_size)
        image_rgb = image.copy()
        image_rgb = cv2.polylines(image_rgb, [self.pts_src], self.isClosed, self.target_color, self.thickness)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = image / 255.0
        image = torch.FloatTensor(image)
        image = image.to(self.DEVICE)
        return image_rgb, image 

    def real_(self):
        video_path = '/home/thomas_yang/ML/hTG_SocialDistance/iphone_video/'
        videos = [video_path + i for i in os.listdir(video_path)]
        videos.sort()
        
        cap = cv2.VideoCapture(videos[3])
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('social_distance_real.avi', fourcc, 30.0, self.model_input_size)
        count = 0
        
        while cap.isOpened(): 
            start_time = time.time()
            ret, frame = cap.read()
            
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            
            count += 1
            if count == 1:
                M, view_land_warp = self.get_view_homography(frame) 
                view_land_warp_last = view_land_warp.copy()
                continue
            view_land_warp_new = view_land_warp.copy()
            
            '''
            preprocess & object-detection 
            '''
            image_rgb, image_nor = self.preprocess(frame)
            detections = self.model(image_nor)[0]
            
            '''
            loop over the detections
            '''
            persons_ori = []
            person_bbox = []
            for i in range(0, len(detections["boxes"])):
            	confidence = detections["scores"][i]
            	if confidence > self.Confidence_thres:
            		idx = int(detections["labels"][i])
            		if idx == 1:
                		box = detections["boxes"][i].detach().cpu().numpy()
                		startX, startY, endX, endY = box.astype("int")        		
                		cv2.rectangle(image_rgb, (startX, startY), (endX, endY), self.COLORS[idx], 1)
                		persons_ori.append([(startX + endX)//2, endY])
                		person_bbox.append([startX, startY, endX, endY])
            persons_ori = np.float32(persons_ori).reshape(-1, 1, 2)
            person_bbox = np.float32(person_bbox)
            persons_warp = cv2.perspectiveTransform(persons_ori, M).astype(np.int32)            
            
            # 
            D = dist.cdist(persons_warp.squeeze(), persons_warp.squeeze(), metric="euclidean")
            for i in range(0, D.shape[0]):
                # center_ori = persons_ori[i].astype(np.int32).squeeze()
                center_warp = persons_warp[i].squeeze()
                center_i = np.array((person_bbox[i][0:2] + person_bbox[i][2:]), dtype=np.int32)//2
                cv2.circle(image_rgb, center_i, 5, self.COLORS[self.color_idx], -1)
                cv2.circle(view_land_warp_new, center_warp, 5, self.COLORS[self.color_idx], -1)
                for j in range(i + 1, D.shape[1]):
                    if D[i, j] < self.MIN_DISTANCE:
                        center_i = persons_warp[i]
                        center_j = persons_warp[j]
                        cv2.line(view_land_warp_new, 
                                 (center_i[0][0], center_i[0][1]),
                                 (center_j[0][0], center_j[0][1]),
                                 (0, 0, 255), 2)
                        center_i = np.array((person_bbox[i][0:2] + person_bbox[i][2:]), dtype=np.int32)//2
                        center_j = np.array((person_bbox[j][0:2] + person_bbox[j][2:]), dtype=np.int32)//2
                        cv2.line(image_rgb, 
                                (center_i[0], center_i[1]),
                                (center_j[0], center_j[1]),
                                (0, 0, 255), 2)
            
            view_land_warp_last[(view_land_warp_new>0) == (view_land_warp_last>0)] = 0
            view_land_warp_new = (view_land_warp_new + view_land_warp_last)
            view_land_warp_last = (view_land_warp_new*0.98).astype(np.uint8)

            cv2.imshow("frame_undistort_c", image_rgb)
            cv2.imshow('view_land_warp_c', view_land_warp_new)
            if cv2.waitKey(1) == ord('q'):
                break
            end_time = time.time()
            print('Frame', count,'process time:', end_time - start_time)
                           
        cap.release()
        out.release()
        cv2.destroyAllWindows()  
        
if __name__ == '__main__':
    pickle_file = 'iphone_calibration.pickle'
    image_path = './simulation_images/img_dist.jpg'
    shift_xy = (400, 600)
    pts_dst = np.array([[0, 0], [30, 0], 
                        [30, 30], [0, 30],
                        ], np.int32) + shift_xy
    sd = SocialDistanceReal(pickle_file, image_path, shift_xy, pts_dst)
    sd.real_()
    
    
    
    
    