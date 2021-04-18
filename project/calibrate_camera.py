#!/usr/bin/env python3

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from p2_functions import calibrate_camera

if __name__ == '__main__':
    ret, mtx, dist, rvecs, tvecs = calibrate_camera('camera_cal/', 9, 6)

    print('mtx:')
    print(mtx)
    print('dist')
    print(dist)

    img = mpimg.imread('test_images/test5.jpg')
    
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    cv2.imwrite('./output_images/undistorted_road.jpg', cv2.cvtColor(undist, cv2.COLOR_RGB2BGR))

    plt.imshow(undist)
    plt.show()