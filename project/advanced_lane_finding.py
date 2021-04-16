#!/usr/bin/env python3

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from p2_functions import calibrate_camera, pipeline


if __name__ == "__main__":
  # Get camera calibration info
  ret, mtx, dist, rvecs, tvecs = calibrate_camera('camera_cal/', 9, 6)
  
  test_images = os.listdir('test_images/') # Get file names for images

  for photo in test_images:
      # Get raw image (still distorted)
      # img_raw = mpimg.imread('test_images/test6.jpg')
      img_raw = mpimg.imread('test_images/' + photo)

      # Undistort the image
      img = cv2.undistort(img_raw, mtx, dist, None, mtx)
      
      # Run undistorted image through pipeline
      result = pipeline(img, mtx, dist, rvecs, tvecs)

      savefile = os.path.splitext(photo)[0]
      savefile += '_out.jpg'

      cv2.imwrite('./output_images/' + savefile, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

  # Compare side by side
  f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
  f.tight_layout()
  ax1.imshow(img)
  ax1.set_title('Before Pipeline', fontsize=50)
  ax2.imshow(result, cmap='gray')
  ax2.set_title('After Pipeline', fontsize=50)
  plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
  plt.show()
