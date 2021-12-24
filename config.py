#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 17:50:38 2021

@author: thomas_yang
"""

import os

classes_label = {"Not": 0,
                 "Raise": 1,}

finetune = True
finetune_load_epoch = 90

batch_size = 32
epochs = 100
learning_rate = 6e-4

get_image_size = (320, 320)
max_boxes_per_image = 100
downsampling_ratio = 4

num_classes = 2 + 1 
tid_classes = 5
# tid_classes = 55

heads = {"heatmap": num_classes, "wh": 2, "reg": 2, "embed": 256,"tid": tid_classes}
bif = 256

hm_weight = 0.001
off_weight = 1.0
wh_weight = 0.1
reid_eright = 0.1

train_data_dir_1 = '/home/thomas_yang/ML/hTG_RaiseHand/txt_file/'
train_data_list_1 = os.listdir(train_data_dir_1)
train_data_list_1.sort()
train_data_list_1 = [train_data_dir_1 + i for i in train_data_list_1]

train_data_dir_2 = '/home/thomas_yang/ML/hTG_MOT_pytorch/txt_file/'
train_data_list_2 = os.listdir(train_data_dir_2)
train_data_list_2.sort()
train_data_list_2 = [train_data_dir_2 + i for i in train_data_list_2]
train_data_list_2 = train_data_list_2[0:19]

train_data_list = train_data_list_1
# train_data_list = train_data_list_1 + train_data_list_2

top_K = 50
score_threshold = 0.2

CLASSES = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush',
    91: 'hair brush'
}

