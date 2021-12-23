#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 12:07:35 2021

@author: thomas_yang
"""

# import config
import cv2
import os
import numpy as np
import sys

video_path = '/home/thomas_yang/ML/hTG_SocialDistance/chessboard_video/'
videos = [video_path + i for i in os.listdir(video_path)]
videos.sort()
image_save_dir = '/home/thomas_yang/ML/hTG_SocialDistance/chessboard_images/'

for videos_name in videos:
    cap = cv2.VideoCapture(videos_name)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    count = 0
    while cap.isOpened():        
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        image_name = videos_name.split('/')[-1].split('.')[0] + ('_%08d.jpg'%count)        
        print(image_name)        
        count+=1
        if count % 30 == 0:
            cv2.imwrite(os.path.join(image_save_dir, image_name), frame)
        
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()
            
cap.release()
cv2.destroyAllWindows()    