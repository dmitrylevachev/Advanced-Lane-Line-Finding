import cv2
import numpy as np
import os
import matplotlib.image as mpimg

class Camera():

    def __init__(self):
        self.mtx = None
        self.dist = None


    def calibrate(self, path):
        file_names = os.listdir(path)

        img_points = []
        obj_points = []
        img_shape = []

        for name in file_names:
            image = mpimg.imread(path + '/' + name)
            obj_p = np.zeros((9*6,3), np.float32)
            obj_p[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            img_shape = gray.shape[::-1]
            ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

            if ret == True:
                img_points.append(corners)
                obj_points.append(obj_p)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_shape, None, None)
        self.mtx = mtx
        self.dist = dist
    
    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, None)

    def warpPerspective(self, img, src, dst):
        m = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img, m, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

def get_transform_points(img):
    offset = 350
    src = np.float32([[579.681/1280*img.shape[1], 458.913/720*img.shape[0]],[703.423/1280*img.shape[1],458.913/720*img.shape[0]],[1114.42/1280*img.shape[1],img.shape[0]],[198.078/1280*img.shape[1], img.shape[0]]])
    dst = np.float32([[0 + offset,0],[img.shape[1] - offset,0],[img.shape[1] - offset,img.shape[0]],[0 + offset,img.shape[0]]])
    return {'src': src, 'dst': dst}