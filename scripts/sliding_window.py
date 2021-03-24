#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from lane_functions import find_lane_pixels, fit_polynomial

binary_warped = mpimg.imread('../images/warped-example.jpg')

out_img = fit_polynomial(binary_warped)

plt.imshow(out_img)
plt.show()
