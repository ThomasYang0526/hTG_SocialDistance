#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 15:08:21 2021

@author: thomas_yang
"""

import torch
import cv2
import numpy as np
from config import CLASSES
from torchvision.models import detection


COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
Confidence_thres = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = {
	"frcnn-resnet": detection.fasterrcnn_resnet50_fpn,
	"frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_320_fpn,
	"retinanet": detection.retinanet_resnet50_fpn
}

model = MODELS["frcnn-resnet"](pretrained=True, progress=True, num_classes=91, pretrained_backbone=True).to(DEVICE)
model.eval()

#%%

    
# from configuration import Config
# import collections
import os
video_path = '/home/thomas_yang/ML/hTG_SocialDistance/iphone_video/'
videos = [video_path + i for i in os.listdir(video_path)]
videos.sort()

cap = cv2.VideoCapture(videos[3])
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('bbox_joint_01.avi', fourcc, 30.0, (960, 540))

while cap.isOpened():    
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # load the image from disk
    image = frame
    image = cv2.resize(image, (480, 270))
    orig = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    image = torch.FloatTensor(image)
    image = image.to(DEVICE)
    detections = model(image)[0]
    
    # loop over the detections
    for i in range(0, len(detections["boxes"])):
    	confidence = detections["scores"][i]
    	if confidence > Confidence_thres:
    		idx = int(detections["labels"][i])
    		if idx == 1:
        		box = detections["boxes"][i].detach().cpu().numpy()
        		startX, startY, endX, endY = box.astype("int")        		
        		cv2.rectangle(orig, (startX, startY), (endX, endY), COLORS[idx], 2)
        		# y = startY - 15 if startY - 15 > 15 else startY + 15        		
        		# label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
        		# cv2.putText(orig, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    cv2.imshow("Output", orig)
    if cv2.waitKey(1) == ord('q'):
        break
                   
cap.release()
out.release()
cv2.destroyAllWindows()





