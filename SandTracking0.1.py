#PlotPy for Falling Sand

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import trackpy as tp
import av

import pandas as pd
from pandas import DataFrame, Series

import pims
from pims import pipeline

print("Enter name of video")

videoName = "/Users/markwang/Documents/LZU/Recordings/" + input()

print("Enter average size of particles in pixels, as an odd number")

particleSize = int(input())

print("Enter the parameter minimum illumination mass to filter detection")

particleTolerance = int(input())

print("Enter the frame to start from")

startFrame = int(input())

#frames is an numpy array of video frames
frames = pims.as_grey(pims.PyAVReaderIndexed(videoName))

#f is the DataFrame of VideoFrames
f = tp.batch(frames[startFrame: startFrame + 20], particleSize, invert = False, minmass = particleTolerance, maxsize = 100, noise_size = 4)

#pred is the prediction algorithm for particles motion assuming new particles are stationay
pred = tp.predict.NearestVelocityPredict()

#Data of particle trajectory
t = pred.link_df(f, 50, memory = 3)


#Display Information
#tp.annotate(f, frames[startFrame])
tp.plot_traj(t)
plt.show()
ax.show()


