#PlotPy for Falling Sand

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

import pims
import av
import trackpy as tp

print("Enter name of video")

videoName = input()

print("Enter average size of particles in pixels, as an odd number")

particleSize = int(input())

print("Enter the parameter mass to filter detection")

particleTolerance = int(input())

#Frames is an numpy array of video frames
frames = pims.Video("/Users/markwang/Documents/LZU/Recordings/" + videoName).to_ndArray()

#F is a list of data for each Frame of Frames
f = []

for frame in frames:
	f.append(tp.locate(frame, particalSize, invert = False, minmass = particalTolerance))

