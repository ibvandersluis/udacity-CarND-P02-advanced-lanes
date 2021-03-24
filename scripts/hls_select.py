#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from lane_functions import hls_select

image = mpimg.imread('../images/test6.jpg')

hls_binary = hls_select(image, thresh=(90, 255))

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(hls_binary, cmap='gray')
ax2.set_title('Thresholded S', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
