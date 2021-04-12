#!/usr/bin/env python3

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from p2_functions import calibrate_camera, pipeline

if __name__ == "__main__":
  # Get camera calibration info
  ret, mtx, dist, rvecs, tvecs = calibrate_camera('camera_cal/', 9, 6)
  
  img = mpimg.imread('test_images/test5.jpg')
  
  result = pipeline(img, mtx, dist, rvecs, tvecs)

  f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
  f.tight_layout()
  ax1.imshow(img)
  ax1.set_title('Before Pipeline', fontsize=50)
  ax2.imshow(result)
  ax2.set_title('After Pipeline', fontsize=50)
  plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
  plt.show()
