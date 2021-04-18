#!/usr/bin/env python3

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from p2_functions import Lane, pipeline


if __name__ == "__main__":
    left_lane = Lane()
    right_lane = Lane()

    test_images = os.listdir('test_images/') # Get file names for images

    for photo in test_images:
        # Get raw image (still distorted)
        img_raw = mpimg.imread('test_images/' + photo)
        
        # Run undistorted image through pipeline
        result = pipeline(img_raw, left_lane, right_lane)

        savefile = os.path.splitext(photo)[0]
        savefile += '_out.jpg'

        cv2.imwrite('./output_images/' + savefile, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    
    # Get raw image (still distorted)
    # img_raw = mpimg.imread('test_images/test5.jpg')

    # # Run undistorted image through pipeline
    # result = pipeline(img_raw, left_lane, right_lane)

    # savefile = os.path.splitext(photo)[0]
    # savefile += '_out.jpg'

    # cv2.imwrite('./output_images/' + savefile, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
