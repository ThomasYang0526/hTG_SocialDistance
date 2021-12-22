import numpy as np
import cv2
import glob
import argparse
import pdb
import pickle
import copy

class CameraCalibration(object):
    def __init__(self, filepath):
        
        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.

        self.cal_path = filepath


    def calibrate_single_camera(self, cam_type, checkerboard_dim = (9,6), checkerboard_size = 0.036):   
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        objp = np.zeros((checkerboard_dim[0]*checkerboard_dim[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_dim[0], 0:checkerboard_dim[1]].T.reshape(-1, 2) * checkerboard_size
        images = glob.glob(self.cal_path + 'Single/Original/{}/*.jpg'.format(cam_type))
        images.sort()

        N_imm = len(images)
        objpoints  = []
        imgpoints = []
        for i, fname in enumerate(images):
            img = cv2.imread(images[i])

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, checkerboard_dim, None)
            
            if ret is True:
                # If found, add object points, image points (after refining them)
                objpoints.append(objp)                
                rt = cv2.cornerSubPix(gray, corners, (11, 11),
                                      (-1, -1), criteria)
                imgpoints.append(corners)

                # Draw and display the corners
                ret = cv2.drawChessboardCorners(img, checkerboard_dim,
                                                  corners, ret)
                cv2.imwrite(self.cal_path + 'Checkerboard/{}/{}.jpg'.format(cam_type,str(i)), img)
                
        img_shape = gray.shape[::-1]
        

        rt, M, d, r, t = cv2.calibrateCamera(
            objpoints, imgpoints, img_shape, None, None)
            
        if cam_type == "LEFT":
            [self.M1, self.d1, self.r1, self.t1] =[ M, d, r, t]
            self.newCameraMatrix_L, self.roi_L = cv2.getOptimalNewCameraMatrix(M, d, img_shape, 1, img_shape)
            
        elif cam_type == "RIGHT":
            [self.M2, self.d2, self.r2, self.t2] = [M, d, r, t]
            self.newCameraMatrix_R, self.roi_R = cv2.getOptimalNewCameraMatrix(M, d, img_shape, 1, img_shape)
        

        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints_1, _ = cv2.projectPoints(objpoints[i], r[i], t[i], M, d)
            error = cv2.norm(imgpoints[i], imgpoints_1, cv2.NORM_L2)/len(imgpoints)
            mean_error += error

        print( cam_type + "total error: {}".format(mean_error/len(objpoints)))
        
        
    def calibrate_single_camera_fisheye(self, cam_type):
        self.calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW

        subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        
        images = glob.glob(self.cal_path + 'Single/Original/{}/*.jpg'.format(cam_type))
        images.sort()
        
        objpoints = []
        imgpoints = []
        _img_shape = None
        N_imm = len(images)
        for fname in images:
            img = cv2.imread(fname)
            if _img_shape == None:
                _img_shape = img.shape[:2]
            else:
                assert _img_shape == img.shape[:2], "All images must share the same size."
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_dim, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(self.objp)
                cv2.cornerSubPix(gray,corners,(3,3),(-1,-1), subpix_criteria)
                imgpoints.append(corners)
            img_shape = gray.shape[::-1]
        
        N_OK = len(objpoints)
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        rt, M, d, r, t = cv2.fisheye.calibrate(
                        objpoints,
                        imgpoints,
                        img_shape,
                        K,
                        D,
                        rvecs,
                        tvecs,
                        self.calibration_flags,
                        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
                    )
            
        if cam_type == "LEFT":
            [self.M1, self.d1, self.r1, self.t1] =[ M, d, r, t]
            

        elif cam_type == "RIGHT":
            [self.M2, self.d2, self.r2, self.t2] = [M, d, r, t]
            

        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints_1, _ = cv2.projectPoints(objpoints[i], r[i], t[i], M, d)
            error = cv2.norm(imgpoints[i], imgpoints_1, cv2.NORM_L2)/len(imgpoints)
            mean_error += error

        print( cam_type + "total error: {}".format(mean_error/len(objpoints)))
    
    def stereo_calibration(self, checkerboard_dim = (7,6), checkerboard_size = 0.06):    
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        
        # calibrate stereo camera

        objp = np.zeros((checkerboard_dim[0]*checkerboard_dim[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_dim[0], 0:checkerboard_dim[1]].T.reshape(-1, 2)*checkerboard_size
        self.objp = objp
        self.checkerboard_dim = checkerboard_dim
        
        images_right = glob.glob(self.cal_path + 'Stereo/Original/RIGHT/*.jpg')
        images_left = glob.glob(self.cal_path + 'Stereo/Original/LEFT/*.jpg')
        images_left.sort()
        images_right.sort()

        
        for i, fname in enumerate(images_right):
            subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
            img_l = cv2.imread(images_left[i])
            img_r = cv2.imread(images_right[i])
            
            #img_l = cv2.undistort(img_l, self.M1, self.d1, None, self.newCameraMatrix_L)
            #img_r = cv2.undistort(img_r, self.M2, self.d2, None, self.newCameraMatrix_R)

            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            #ret_l, corners_l = cv2.findChessboardCorners(gray_l, checkerboard_dim, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
            #ret_r, corners_r = cv2.findChessboardCorners(gray_r, checkerboard_dim, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
            
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, checkerboard_dim)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, checkerboard_dim)
            
            #corners_l_ = copy.copy(corners_l)
            #corners_l_ = copy.copy(corners_l)

            if ret_l is True and ret_r is True:
                # If found, add object points, image points (after refining them)
                self.objpoints.append(objp)
                
                rt_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11),
                                      (-1, -1), subpix_criteria)
                self.imgpoints_l.append(corners_l)

                # Draw and display the corners
                ret_l = cv2.drawChessboardCorners(img_l, checkerboard_dim,
                                                  corners_l, ret_l)
                

                rt_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11),
                                      (-1, -1), subpix_criteria)
                self.imgpoints_r.append(corners_r)

                # Draw and display the corners
                ret_r = cv2.drawChessboardCorners(img_r, checkerboard_dim,
                                                  corners_r, ret_r)
                
                cv2.imwrite(self.cal_path + 'Stereo/Checkerboard/{}/{}.jpg'.format('LEFT',str(i)), ret_l)
                cv2.imwrite(self.cal_path + 'Stereo/Checkerboard/{}/{}.jpg'.format('RIGHT',str(i)), ret_r)
            img_shape = gray_l.shape[::-1]
            
        #pdb.set_trace()
        self.camera_model = self.stereo_calibrate(img_shape)

    def stereo_calibrate(self, dims):
        
        flags = 0
        
        flags |= cv2.CALIB_FIX_INTRINSIC        
        #flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        #flags |= cv2.CALIB_FIX_FOCAL_LENGTH        
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
        
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO

        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                                cv2.TERM_CRITERIA_EPS, 30, 0.001)
        

        
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l,
            self.imgpoints_r, self.M1, self.d1, self.M2,
            self.d2, dims,
            None, None, None, None,
            criteria=stereocalib_criteria, flags=flags)

        print('rmse:', ret)
        print('Intrinsic_mtx_1', M1)
        print('dist_1', d1)
        print('Intrinsic_mtx_2', M2)
        print('dist_2', d2)
        print('R', R)
        print('T', T)
        print('E', E)
        print('F', F)

        
        # for i in range(len(self.r1)):
        #     print("--- pose[", i+1, "] ---")
        #     self.ext1, _ = cv2.Rodrigues(self.r1[i])
        #     self.ext2, _ = cv2.Rodrigues(self.r2[i])
        #     print('Ext1', self.ext1)
        #     print('Ext2', self.ext2)

        print('')
        #pdb.set_trace()
        camera_model = dict([('M1', M1), ('M2', M2), ('dist1', d1),
                            ('dist2', d2), ('rvecs1', self.r1),
                            ('rvecs2', self.r2), ('R', R), ('T', T),
                            ('E', E), ('F', F), ('rmse', ret)])
        
        #p_2 = R * p_1 + T 

        cv2.destroyAllWindows()
        return camera_model

if __name__ == '__main__':
    path = './images/'
    stereo_checkerboard = (9,6)
    stereocalibration = CameraCalibration(path)
    stereocalibration.calibrate_single_camera('LEFT')
    stereocalibration.calibrate_single_camera('RIGHT')
    stereocalibration.stereo_calibration(checkerboard_dim = stereo_checkerboard, checkerboard_size = 0.036)
    cal_data = stereocalibration.camera_model
    
    with open('stereo_calibration.pickle', 'wb') as handle:
        pickle.dump(cal_data, handle)
    
    
    img_l = cv2.imread("./images/Single/Original/LEFT/0.jpg")

    h_l,  w_l = img_l.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cal_data["M1"], cal_data["dist1"], (w_l,h_l), 1, (w_l,h_l))
    dst_l = cv2.undistort(img_l, cal_data["M1"], cal_data["dist1"], None, newcameramtx)
    cv2.imshow("Left0", dst_l)
        
    img_r = cv2.imread("./images/Single/Original/RIGHT/10.jpg")
    h_r,  w_r = img_r.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cal_data["M2"], cal_data["dist2"], (w_r,h_r), 1, (w_r,h_r))
    dst_r = cv2.undistort(img_r, cal_data["M2"], cal_data["dist2"], None, newcameramtx)
    cv2.imshow("Right0", dst_r)
    
    
    
    # Stereo
    target = 0
    img_l = cv2.imread("./images/Stereo/Original/LEFT/{}.jpg".format(str(target)))
    h_l,  w_l = img_l.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cal_data["M1"], cal_data["dist1"], (w_l,h_l), 1, (w_l,h_l))
    dst_l = cv2.undistort(img_l, cal_data["M1"], cal_data["dist1"], None, newcameramtx)
    cv2.imshow("Left", dst_l)
    
    img_r = cv2.imread("./images/Stereo/Original/RIGHT/{}.jpg".format(str(target)))
    h_r,  w_r = img_r.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cal_data["M1"], cal_data["dist1"], (w_r,h_r), 1, (w_r,h_r))
    dst_r = cv2.undistort(img_r, cal_data["M1"], cal_data["dist1"], None, newcameramtx)
    cv2.imshow("Right", dst_r)
    
    objpoints = stereocalibration.objp
    gray = cv2.cvtColor(img_l ,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, stereo_checkerboard,None)
    ret,rvecs, tvecs = cv2.solvePnP(objpoints, corners, cal_data["M1"] , cal_data["dist1"])
    rot_matrix = cv2.Rodrigues(rvecs)[0]
    transformation1 = np.zeros((4,4))
    transformation1[:3,:3] = rot_matrix
    transformation1[:3,-1] = np.squeeze(tvecs)
    transformation1[-1,-1] = 1
    
    transformation2 = np.zeros((4,4))
    transformation2[:3,:3] = cal_data["R"]
    transformation2[:3,-1] = np.squeeze(cal_data["T"])
    transformation2[-1,-1] = 1
    
    transformation3 = np.matmul(transformation2, transformation1)
    
    
    gray = cv2.cvtColor(img_r ,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, stereo_checkerboard,None)
    ret,rvecs, tvecs = cv2.solvePnP(objpoints, corners, cal_data["M2"] , cal_data["dist2"])
    rot_matrix = cv2.Rodrigues(rvecs)[0]
    #objpoints_1 = (np.matmul(rot_matrix , objpoints.T) + tvecs).T
    #objpoints_2 = (np.matmul(cal_data["R"] , objpoints.T) + cal_data["T"] ).T
    #r = cv2.Rodrigues(np.eye(3))[0]
    r = transformation3[:3,:3]
    t = transformation3[:3,-1]
    imgpoints_1, _ = cv2.projectPoints(objpoints, r, t, cal_data["M2"] , cal_data["dist2"])
    ret = cv2.drawChessboardCorners(img_r, stereocalibration.checkerboard_dim, imgpoints_1, ret)
    cv2.imshow("Stereo_2", img_r)
    #cv2.imwrite("Stereo2.jpg", img_r)
    print(cal_data['rmse'])
    
