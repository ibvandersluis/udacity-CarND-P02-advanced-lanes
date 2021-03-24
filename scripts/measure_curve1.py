#!/usr/bin/env python3

from lane_functions import generate_data, measure_curvature_pixels


left_curverad, right_curverad = measure_curvature_pixels()

print(left_curverad, right_curverad)
