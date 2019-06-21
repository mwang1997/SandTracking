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
f = tp.batch(frames[startFrame: startFrame + 50], particleSize, invert = False, minmass = particleTolerance)

#pred is the prediction algorithm for particles motion assuming new particles are stationay
pred = tp.predict.NearestVelocityPredict()

#Data of particle trajectory
t = pred.link_df_iter(f, memory = 3)


tp.annotate(f, frames[1000])
fig, ax = plt.subplots()
ax.hist(f['mass'], bins=1000)
plt.show()
ax.show()


