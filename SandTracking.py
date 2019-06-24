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
		self.irregular = 0

		#Tuples of position and its 1st, 2nd and 3rd derivatives
		self.coords = []
		self.vel = []
		self.accel = []
		self.jerk = []
		
		#Bookkeeping information with index 0 of average being velocity, etc etc
		self.average = []
		self.index = []

	def add_index(self, coord, diameter, ecc, index):
		self.diameter = (self.diameter * len(self.coords) + diameter) / (len(self.coords) + 1)
		self.ecc = (self.ecc * len(self.coords) + ecc) / (len(self.coords) + 1)
		self.used_index.append(index)
		self.index.append(index)
		self.coords.append(coord)

	def calc_vel(self):
		self.average.append((0, 0))

		#Goes through all the coordinates and calculates velocity
		for i in range (0, len(self.coords) - 1):
			#The time that the velocity is calculated at, with FPS
			dt = (self.coords[i + 1][2] - self.coords[i][2]) / 1000
			t = (self.coords[i + 1][2] + self.coords[i][2]) / 2

			self.vel.append(((self.coords[i + 1][0] - self.coords[i][0]) / dt, (self.coords[i + 1][1] - self.coords[i][1]) / dt, t))

			#Filters velocity direction from changing
			if self.vel[-1][0] * self.average[-1][0] < 0 or self.vel[-1][1] * self.average[-1][1] < 0:
				self.irregular += 1

			self.average[-1]  = (self.average[-1][0] + self.vel[-1][0], self.average[-1][1] + self.vel[-1][1])

		#Used to filter out "particles" that are not moving
		self.average[-1] = (self.average[-1][0] / len(self.vel), self.average[-1][1] / len(self.vel))

	def calc_accel(self):
		self.average.append((0, 0))


		#Goes through all the velocities and calculates drag
		for i in range (0, len(self.vel) - 1):
			#The time that the velocity is calculated at, with FPS
			dt = (self.vel[i + 1][2] - self.vel[i][2]) / 1000
			t = (self.vel[i + 1][2] + self.vel[i][2]) / 2

			self.accel.append(((self.vel[i + 1][0] - self.vel[i][0]) / dt, (self.vel[i + 1][1] - self.vel[i][1]) / dt, t))
			self.average[-1]  = (self.average[-1][0] + self.accel[-1][0], self.average[-1][1] + self.accel[-1][1])

		#Used to filter out particles that are not dropping down due to gravity
		self.average[-1] = (self.average[-1][0] / len(self.accel), self.average[-1][1] / len(self.accel))

	def calc_jerk(self):
		self.average.append((0, 0))

		#Goes through all the coordinates and 
		for i in range (0, len(self.accel) - 1):
			#The time that the velocity is calculated at, with FPS
			dt = (self.accel[i + 1][2] - self.accel[i][2]) / 1000
			t = (self.accel[i + 1][2] + self.accel[i][2]) / 2

			self.jerk.append(((self.accel[i + 1][0] - self.accel[i][0]) / dt, (self.accel[i + 1][1] - self.accel[i][1]) / dt, t))
			self.average[-1]  = (self.average[-1][0] + self.jerk[-1][0], self.average[-1][1] + self.jerk[-1][1])

		#Used to filter out particles that are not dropping down due to gravity
		self.average[-1] = (self.average[-1][0] / len(self.jerk), self.average[-1][1] / len(self.jerk))

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

		particles[ID].add_index((traj.at[i, "x"], traj.at[i, "y"], traj.at[i, "frame"]), traj.at[i, "size"] * 2, traj.at[i, "ecc"], i)

	return particles

#filter stub that doesn't delete the index 
def better_filter_stubs(data_frame, i):
	t = tp.filter_stubs(data_frame, i)
	t.index = range(0, len(t))

	return t

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
f = tp.batch(frames[startFrame: startFrame + 50], particleSize, invert = False, minmass = particleTolerance, noise_size = 4)

#pred is the prediction algorithm for particles motion assuming new particles are stationay
pred = tp.predict.NearestVelocityPredict()

#Data of particle trajectory, in DataFrame and Dictionary form
t = pred.link_df(f, 50, memory = 1)
t = better_filter_stubs(t, 10)

print(t)

#Converted to particles containing their trajectories
particle_dict = extractParticles(t).copy()

#Caclulates all necessary parameters
for p in particle_dict.values():
	p.analyze()

#Post Filtering
for p in particle_dict.copy().values():
	#If the particle is effectively stationary or the particles experience collision
	if (p.average[0][1] < 1000 and p.average[0][1] < 1000) or p.irregular > 0:
		particle_dict.pop(p.ID)

		#remove from 
		for i in p.index:
			particle.used_index.remove(i)

t = t.loc[particle.used_index]

for p in particle_dict.values():
	print(p.average[0])
	print(p.average[1])
	print(p.average[2])

print(t)

plt.figure()
tp.plot_traj(t)

