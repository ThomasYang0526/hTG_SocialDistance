#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 11:19:53 2021

@author: thomas_yang
"""

import pickle
from camera_calibration import CameraCalibration

if __name__ == '__main__':
    path = './chessboard_images/'
    stereo_checkerboard = (9,6)
    stereocalibration = CameraCalibration(path)
    stereocalibration.calibrate_single_camera('LEFT')
    
    camera_model = dict([('M1', stereocalibration.M1), 
                         ('dist1', stereocalibration.d1),
                         ('r1', stereocalibration.r1),
                         ('t1', stereocalibration.t1),
                         ])
    
    with open('iphone_calibration.pickle', 'wb') as handle:
        pickle.dump(camera_model, handle)    