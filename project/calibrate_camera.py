#!/usr/bin/env python3

from p2_functions import calibrate_camera

if __name__ == '__main__':
    ret, mtx, dist, rvecs, tvecs = calibrate_camera('camera_cal/', 9, 6)

    print('mtx:')
    print(mtx)
    print('dist')
    print(dist)