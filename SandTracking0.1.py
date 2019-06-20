#PlotPy for Falling Sand

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import av
import trackpy as tp

import pandas as pd
from pandas import DataFrame, Series

import pims
from pims import pipeline

#Transforms list of frames in numpy array format into greyscale
def to_greyscale(frame):
	red = frame[:, :, 0]
	green = frame[:, :, 1]
	blue = frame[:, :, 2]
	return 0.2126 * red + 0.7152 * green + 0.0722 * blue

print("Enter name of video")

videoName = "/Users/markwang/Documents/LZU/Recordings/" + input()

print("Enter average size of particles in pixels, as an odd number")

particleSize = int(input())

print("Enter the parameter mass to filter detection")

particleTolerance = int(input())

#Frames is an numpy array of video frames
frames = []
container = av.open(videoName)
for frame in container.decode(video = 0):
	frames.append(frame.to_ndarray())

plt.imshow(frames[0])
plt.show()

#f is a list of data for each Frame of Frames
f = []

for frame in frames:
	f.append(tp.locate(frame, particleSize, invert = False, minmass = particleTolerance))


