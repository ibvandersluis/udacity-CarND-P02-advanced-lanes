#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from lane_functions import hist

# `mpimg.imread` will load .jpg as 0-255, so normalize back to 0-1
img = mpimg.imread('../images/warped-example.jpg')/255

# Create histogram of image binary activations
histogram = hist(img)

# Visualize the resulting histogram
plt.plot(histogram)
plt.show()
