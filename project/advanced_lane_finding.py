#!/usr/bin/env python3

from moviepy.editor import VideoFileClip
from p2_functions import Lane, pipeline

if __name__ == '__main__':
    left_lane = Lane()
    right_lane = Lane()

    output = 'output_video.mp4'
    clip = VideoFileClip('project_video.mp4')
    output_clip = clip.fl_image(lambda image: pipeline(image, left_lane, right_lane))

    output_clip.write_videofile(output, audio=False)