#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from lane_functions import abs_sobel_thresh 

image = mpimg.imread('../images/signs_vehicles_xygrad.png')

grad_binary = abs_sobel_thresh(image, orient='x', thresh=(20, 100))
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(grad_binary, cmap='gray')
ax2.set_title('Thresholded Gradient', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
