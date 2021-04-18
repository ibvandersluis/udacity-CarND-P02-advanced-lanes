#!/usr/bin/env python3

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from p2_functions import calibrate_camera

if __name__ == "__main__":
    ret, mtx, dist, rvecs, tvecs = calibrate_camera('camera_cal/', 9, 6)

    img_raw = mpimg.imread('test_images/straight_lines1.jpg')

    img = cv2.undistort(img_raw, mtx, dist, None, mtx)

    img_size = (img.shape[1], img.shape[0])

    # Set the source points for the transformation
    src = np.float32([[190, 720],
                      [593, 450],
                      [687, 450],
                      [1090, 720]])

    # Set the destination points for the transformation
    dst = np.float32([[260, 720],
                      [260, 0],
                      [1020, 0],
                      [1020, 720]])

    # Get the tranformation matrix from src to destination
    tf_M = cv2.getPerspectiveTransform(src, dst)

    # Tranform the perspective of the image to top-down
    tf_img = cv2.warpPerspective(img, tf_M, img_size)

    # Draw rectangles
    img = cv2.polylines(img, np.int32([src]), True, (255,0,0), 4)

    tf_img = cv2.polylines(tf_img, np.int32([dst]), True, (255,0,0), 4)

    plt.imshow(img)
    cv2.imwrite('./output_images/before_tf.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    plt.show()

    plt.imshow(tf_img)
    cv2.imwrite('./output_images/after_tf.jpg', cv2.cvtColor(tf_img, cv2.COLOR_RGB2BGR))
    plt.show()

    # Compare side by side
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    # f.tight_layout()
    # ax1.imshow(img)
    # ax1.set_title('Original', fontsize=50)
    # ax2.imshow(tf_img)
    # ax2.set_title('Transformed', fontsize=50)
    # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    # plt.show()