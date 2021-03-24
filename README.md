# Lane Detection

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

A repository for the lane detection exercises and project from Udacity's Self-Driving Car Engineer nanodegree. (Project #1)

The goal of this project is to take an image of the road and build a pipeline to detect the lanes in it. After tested on several images, this pipeline in then applied to images in a video.

I will start with an image of the road like this:

![Image before pipeline](./project/test_images/solidWhiteCurve.jpg)

And after the pipeline I will produce an image like this, with annotade lane markings:

![Image after pipeline](./project/test_images_output/solidWhiteCurve_out.jpg)

Some basic scripts from the lessons and exercises are located in [`scripts/`](./scripts). The images used for those scripts are found in [`images/`](./scripts). The project, started from the project Udacity repository [here](https://github.com/udacity/CarND-LaneLines-P1), is found in [`project/`](./project).

## Launching the Project

Navigate to the `project/` directory and run

```bash
jupyter notebook
```

Then open P1.ipynb.

Find the writeup for the project [here](./project/writeup.md).
