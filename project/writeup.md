**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./camera_cal/calibration1.jpg "Distorted"
[image1]: ./output_images/undistorted.jpg "Undistorted Chessboard"
[image2]: ./output_images/undistorted_road.jpg "Undistorted Road"
[image3]: ./output_images/step1.jpg "Binary Example"
[image4]: ./output_images/step2.jpg "Warp Example"
[image5]: ./output_images/step3.jpg "Find Lane Lines"
[image6]: ./output_images/test5_out.jpg "Output"
[image7]: ./output_images/before_tf.jpg "Before Transform"
[image8]: ./output_images/after_tf.jpg "After Transform"
[video1]: ./output_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in `calibrate_camera()` between lines 31 and 65 in `p2_functions.py`.

I start by getting the file names for every calibration image I will use. Before I iterate through these images, I declare empty lists for my object points (the 3D coordinates of the chessboard corners on a flat plane), and image points (which will hold the actual detected 2D coordinates of the corners in the calibration image). These image and object points will be appended to their respective lists after the corners in the calibration image have been found, but not all calibration images have fully visible corners. Therefore the points are only appended if the chessboard corners are successfully found.

Once all calibration images have been inspected and their image and object points appended where appropriate, the use these points and the image shape to calibrate the camera. In `calibrate_camera.py`, I output the camera and distortion matrices so that I can hard-code them into my pipeline (line 369-372 of `p2_functions.py`)

The resulting matrices are as follows:

```python
mtx = np.array([[1.15777942e+03, 0.00000000e+00, 6.67111050e+02],
                [0.00000000e+00, 1.15282305e+03, 3.86129068e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist = np.array([[-0.24688833, -0.02372817, -0.00109843,  0.00035105, -0.00259134]])
```
Before undistortion:
![alt text][image0]
After undistortion:
![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The undistortion, using the camera and distortion matrices mentioned above, looks like this when applied to an image of the road:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

In the HLS colour space, I used a combination of thresholds from the L (light) and S (saturation) channels to produce my binary images. The steps for this are in the `thresholds()` function in `p2_functions.py`, lines 346-367. From the L channel I take the gradient in the x direction with the thresholds `(20, 100)`. From the S channel I apply colour selection with  thresholds `(170, 255)`.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform was tested in the script `transform.py` before being copied over to lines 383-409 in `pipeline()` from `p2_functions.py`. I hand-picked my `src` and `dst` points for the tranformation by making sure the `src` points were equidistant from the middle of the undistorted image in the x-direction (midline at 640 px) and roughly approximated the lane lines in the images. The top points are +/- 47 px from the middle, while the bottom points are +/- 450 px from the middle. The `dst` points were selected to form a wide rectangle that maximised lane area and minimised interference from objects or boundaries outside of the lane This rectangle was centred in the frame.

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 190, 720      | 260, 720      | 
| 593, 450      | 260, 0        |
| 687, 450      | 1020, 0       |
| 1090, 720     | 1020, 720     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear roughly parallel in the warped image.

Before transform:
![alt text][image7]
After transform:
![alt text][image8]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used the sliding-window technique to identify lane pixels in the frames. By taking a histogram of the bottom half of the transofmed binary and identifying the maximums in the right and left sections, I designated the starting positions for the windows. The number of windows from top to bottom is 9, the margin (width) for each window is 100 px to each side, and the minimum number of pixels required to recentre the window is 50. Once the lane pixels have been detected, they are fit with a second-order polynomial to approximate the lane boundaries.

The result looks like this:

![alt text][image5]

The code for this is found in `find_lane_pixels()` and `fit_polynomial()` in `p2_functions.py` (lines 124-240).

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Using the knowledge that lanes are 3.7 metres wide and dashed lanes are 3 metres long, I manually measure these lengths from my perspective-transformed image. Since a dashed lane is 80 px long in the y-direction, I use:

```python
ym_per_pix = 3/80.0
```
And since the width of the lane is about 736 px (measured from the inside edge of each lane), I use:

```python
xm_per_pix = 3.7/736
```

I then make my calculates in `measure_curvature()` and `measure_position()` in `p2_functions.py` (lines 311-348).

NOTE: A negative distance from the centre of the lane indicates that the vehicle is left-of-centre, while a positive distance indicates that the vehicle is right-of-centre

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I reverse the perspective transform and overlay the annotations on the undistorted image in lines 421 and 424 of `p2_functions.py` (in the `pipeline()` function). 

Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My pipeline can fail if lines parallel to the lane lines are detected after the perspective transform, causing the window searching to start int the wrong place. The can be fixed with smoothing and a sanity check, where previous values are used if the current ones seem dubious. This could also be remedied through further tweaking of the thresholding parameters, eliminating such errors as mistaking a shadow for a lane line.

There is also some jitter at certain places in the video. The could also be fixed with smoothing and sanity checks.
