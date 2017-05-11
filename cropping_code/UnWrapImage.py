import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt


def undistort_image(image):
    width = image.shape[1]
    height = image.shape[0]
    distCoeff = np.zeros((4, 1), np.float64)
    k1 = -6.0e-5  # negative to remove barrel distortion
    k2 = 0.0
    p1 = 0.0
    p2 = 0.0
    distCoeff[0, 0] = k1
    distCoeff[1, 0] = k2
    distCoeff[2, 0] = p1
    distCoeff[3, 0] = p2
    # assume unit matrix for camera
    cam = np.eye(3, dtype=np.float32)
    cam[0, 2] = width / 2.0  # define center x
    cam[1, 2] = height / 2.0  # define center y
    cam[0, 0] = 10.  # define focal length x
    cam[1, 1] = 10.  # define focal length y
    # here the undistortion will be computed
    return cv2.undistort(image, cam, distCoeff)


def UnWrapImage():
    for index in range(0, 12):
        imgpath = './ShelfB/bottles_B' + str(index) + '.png'
        img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
        unwarpimg = undistort_image(image=img)
        unimg = './UnShelfB/bottles_B' + str(index) + '.png'
        cv2.imwrite(unimg, unwarpimg)
    # cv2.imshow('unwarpimg', unwarpimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
