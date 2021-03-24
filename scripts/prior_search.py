#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from lane_functions import fit_poly, search_around_poly

# Load our image - this should be a new frame since last time!
binary_warped = mpimg.imread('../images/warped-example.jpg')

# Polynomial fit values from the previous frame
# Make sure to grab the actual values from the previous step in your project!
left_fit = np.array([ 2.13935315e-04, -3.77507980e-01,  4.76902175e+02])
right_fit = np.array([4.17622148e-04, -4.93848953e-01,  1.11806170e+03])

# Run image through the pipeline
# Note that in your project, you'll also want to feed in the previous fits
result = search_around_poly(binary_warped, left_fit, right_fit)

# View your output
plt.imshow(result)
plt.show()
