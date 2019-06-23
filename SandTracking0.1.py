#PlotPy for Falling Sand

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import trackpy as tp

import av
import pims

import pandas as pd
from pandas import DataFrame, Series

#All the x and y coordinate values of a particle with their frame and ID
class particle:
	used_index = []

	def __init__(self, ID):
		self.ID = ID
		self.diameter = 0
		self.ecc = 0

		#Tuples of 0th, 1st, 2nd, 3rd derivatives of position
		self.coords = []
		self.vel = []
		self.accel = []
		self.jerk = []
		self.index = []

		self.average_y_vel = 0

	def add_frame(self, coord, diameter, ecc, index):
		self.diameter = (self.diameter * len(self.coords) + diameter) / (len(self.coords) + 1)
		self.ecc = (self.ecc * len(self.coords) + ecc) / (len(self.coords) + 1)
		self.used_index.append(index)
		self.index.append(index)
		self.coords.append(coord)

	def calc_vel(self):
		#Goes through all the coordinates and calculates velocity
		for i in range (0, len(self.coords) - 1):
			#The time that the velocity is calculated at, with FPS
			dt = (self.coords[i + 1][2] - self.coords[i][2]) / 1000
			t = (self.coords[i + 1][2] + self.coords[i][2]) / 2

			self.vel.append(((self.coords[i + 1][0] - self.coords[i][0]) / dt, (self.coords[i + 1][1] - self.coords[i][1]) / dt, t))
			self.average_y_vel  += (self.coords[i + 1][1] - self.coords[i][1]) / dt

		self.average_y_vel /= len(self.coords)

	def calc_accel(self):
		#Goes through all the velocities and calculates drag
		for i in range (0, len(self.vel) - 1):
			#The time that the velocity is calculated at, with FPS
			dt = (self.vel[i + 1][2] - self.vel[i][2]) / 1000
			t = (self.vel[i + 1][2] + self.vel[i][2]) / 2

			self.accel.append(((self.vel[i + 1][0] - self.vel[i][0]) / dt, (self.vel[i + 1][1] - self.vel[i][1]) / dt, t))

	def calc_jerk(self):
		#Goes through all the coordinates and 
		for i in range (0, len(self.accel) - 1):
			#The time that the velocity is calculated at, with FPS
			dt = (self.accel[i + 1][2] - self.accel[i][2]) / 1000
			t = (self.vel[i + 1][2] + self.vel[i][2]) / 2

			self.jerk.append(((self.accel[i + 1][0] - self.accel[i][0]) / dt, (self.accel[i + 1][1] - self.accel[i][1]) / dt, t))

	def analyze(self):
		self.calc_vel()
		self.calc_accel()
		self.calc_jerk()

#Returns a Dictionary containing a list of particle motion from a trajectory DataFrame
def extractParticles(traj):
	#return dictionary
	particles = dict()
	print(traj.shape[0])

	for i in range (0, traj.shape[0]):
		#ID of Particle
		ID = traj.at[i, "particle"]

		if not ID in particles:
			particles[ID] = particle(ID)

		particles[ID].add_frame((traj.at[i, "x"], traj.at[i, "y"], traj.at[i, "frame"]), traj.at[i, "size"] * 2, traj.at[i, "ecc"], i)

	return particles


print("Enter name of video")
videoName = "Recordings/" + input()

print("Enter average size of particles in pixels, as an odd number")
particleSize = int(input())

print("Enter the parameter minimum illumination mass to filter detection")
particleTolerance = int(input())

print("Enter the frame to start from")
startFrame = int(input())

#frames is an numpy array of video frames
frames = pims.as_grey(pims.PyAVReaderIndexed(videoName))

#f is the DataFrame of VideoFrames
f = tp.batch(frames[startFrame: startFrame + 30], particleSize, invert = False, minmass = particleTolerance, noise_size = 4)

#pred is the prediction algorithm for particles motion assuming new particles are stationay
pred = tp.predict.NearestVelocityPredict()

#Data of particle trajectory, in DataFrame and Dictionary form
t = pred.link_df(f, 50, memory = 1)

#Converted to particles containing their trajectories
particle_dict = extractParticles(t).copy()

#Caclulates all necessary parameters
for p in particle_dict.values():
	p.analyze()
	for c in p.coords:
		print(str(c) + " " + str(p.ID))

#Post Filtering
for p in particle_dict.copy().values():
	#If the y velocity is not low enough or the particles do not appear enough
	if p.average_y_vel > 500 or len(p.coords) < 10:
		particle_dict.pop(p.ID)

		#remove from 
		for i in p.index:
			particle.used_index.remove(i)

t = t.loc[particle.used_index]

print(t)

plt.figure()
tp.plot_traj(t)

