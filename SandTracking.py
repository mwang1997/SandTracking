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
		self.irregular = []

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
		self.index.append(index)
		self.coords.append(coord)

		#Checks to not double add
		if not index in self.used_index:
			self.used_index.append(index)

	def calc_vel(self):
		self.average.append((0, 0))

		#Goes through all the coordinates and calculates velocity
		for i in range (0, len(self.coords) - 1):
			#The time that the velocity is calculated at, with FPS
			dt = (self.coords[i + 1][2] - self.coords[i][2]) / 1000
			t = (self.coords[i + 1][2] + self.coords[i][2]) / 2

			self.vel.append(((self.coords[i + 1][0] - self.coords[i][0]) / dt, (self.coords[i + 1][1] - self.coords[i][1]) / dt, t))

			#Filters velocity direction from changing
			if len(self.vel) > 1 and (self.vel[-1][0] * self.vel[-2][0] < 0 or self.vel[-1][1] * self.vel[-2][1] < 0):
				self.irregular.append(self.index[i])

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
			self.average[-1]  = ((self.average[-1][0] * len(self.accel) + self.accel[-1][0]) / (len(self.accel) + 1), 
								(self.average[-1][1] * len(self.accel) + self.accel[-1][1]) / (len(self.accel) + 1))

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

@pims.pipeline
def to_grey(frame):
        red = frame[:, :, 0]
        blue = frame[:, :, 1]
        green = frame[:, :, 2]

        return red * 0.2125 + 0.7154 * green + 0.0721 * blue

def evaluate_features(video_name, particle_size, particle_tolerance, start_frame, frame_length):
	#frames is an numpy array of video frames
	frames = to_grey(pims.PyAVReaderTimed(video_name))

	#f is the DataFrame of VideoFrames
	f = tp.batch(frames[start_frame: start_frame + frame_length], particle_size, minmass = particle_tolerance, noise_size = 4)

	return f

def evaluate_trajectories(data_frame, search_size, lb_search_size, step, particle_memory):
	#pred is the prediction algorithm for particles motion assuming new particles are stationay
	pred = tp.predict.NearestVelocityPredict()
	#Gets the DataFrame of the trajectories
				
	t = pred.link_df(data_frame, search_size, adaptive_stop = lb_search_size, adaptive_step = step, memory = particle_memory)

	return t

#Returns a Dictionary containing a list of particle motion from a trajectory DataFrame
def extract_particles(traj):
	#return dictionary
	particles = dict()

	for i in range (0, traj.shape[0]):
		#ID of Particle
		ID = traj.at[i, "particle"]

		if not ID in particles:
			particles[ID] = particle(ID)

		particles[ID].add_index((traj.at[i, "x"], traj.at[i, "y"], traj.at[i, "frame"]), traj.at[i, "size"] * 2, traj.at[i, "ecc"], i)

	return particles

#Splits a single particle and returns another particle
def split(data_frame, p, filter_stub):
		return_particles = []
		for x in range(0, len(p.irregular) + 1):
			return_particles.append(particle(data_frame.max()[9] + 1 + x))

		return_index = 0
		for x in p.index:
			#if you've reached the irregularity index
			if x in p.irregular:
				return_index += 1

			return_particles[return_index].add_index((data_frame.at[x, "x"], data_frame.at[x, "y"], data_frame.at[x, "frame"]), 
													data_frame.at[x, "size"] * 2, data_frame.at[x, "ecc"], x)

		for rp in return_particles.copy():
			if len(rp.coords) < filter_stub:
				for x in range(0, len(return_particles)):
					if return_particles[x].ID == rp.ID:
						return_particles.pop(x)
						break;
				for x in rp.index:
					particle.used_index.remove(x)

		for rp in return_particles:
			rp.analyze()
			for x in rp.index:
				data_frame.at[x, "particle"] = rp.ID

		data_frame = data_frame.loc[particle.used_index]
			
		return return_particles

def unfilter_jumps(data_frame, particles, filter_stub):
	#For all the particles
	for p in particles.copy().values():
		if not len(p.irregular) == 0:
			new_particles = split(data_frame, p, filter_stub)
			particles.pop(p.ID)
			for split_particles in new_particles:
				particles[split_particles.ID] = split_particles

	return data_frame.loc[particle.used_index]

#filter stub that doesn't delete the index 
def fixed_filter_stubs(data_frame, i):
	t = tp.filter_stubs(data_frame, i)
	t.index = range(0, len(t))

	return t

def postfiltering(data_frame, particles, stillness, tolerance):
	#Post Filtering
	for p in particles.copy().values():
		p.analyze()

		#If the particle is effectively stationary or the particles experience collision
		if (abs(p.average[0][1]) < stillness and abs(p.average[0][1] < stillness)) or len(p.irregular) > tolerance:
			particles.pop(p.ID)

			#remove from 
			for i in p.index:
				particle.used_index.remove(i)

	data_frame = data_frame.loc[particle.used_index]

	return data_frame

def export(data_frame, particles):
	#data to turn into excel sheets
	raw_data = data_frame.copy()
	velocity_data = dict({"x_vel": [], "y_vel": [], "frame": [], "particle": []})
	acceleration_data = dict({"x_accel": [], "y_accel": [], "frame": [], "particle": []})
	jerk_data = dict({"x_jerk": [], "y_jerk": [], "frame": [], "particle": []})

	for p in particles.values():
		for i in range (0, len(p.vel)):
			velocity_data["x_vel"].append(p.vel[i][0])
			velocity_data["y_vel"].append(p.vel[i][1])
			velocity_data["frame"].append(p.vel[i][2])
			velocity_data["particle"].append(p.ID)

			if i < len(p.accel):
				acceleration_data["x_accel"].append(p.accel[i][0])
				acceleration_data["y_accel"].append(p.accel[i][1])
				acceleration_data["frame"].append(p.accel[i][2])
				acceleration_data["particle"].append(p.ID)

			if i < len(p.jerk):
				jerk_data["x_jerk"].append(p.jerk[i][0])
				jerk_data["y_jerk"].append(p.jerk[i][1])
				jerk_data["frame"].append(p.jerk[i][2])
				jerk_data["particle"].append(p.ID)

	velocity_data = pd.DataFrame.from_dict(velocity_data)
	acceleration_data = pd.DataFrame.from_dict(acceleration_data)
	jerk_data = pd.DataFrame.from_dict(jerk_data)

	with pd.ExcelWriter("output.xlsx") as writer:
		raw_data.to_excel(writer, sheet_name = "raw data")
		velocity_data.to_excel(writer, sheet_name = "velocity data")
		acceleration_data.to_excel(writer, sheet_name = "acceleration data")
		jerk_data.to_excel(writer, sheet_name = "jerk data")

##Execution##

print("Enter name of video")
videoName = "Recordings/" + input()

print("Enter average size of particles in pixels, as an odd number")
particleSize = int(input())

print("Enter the parameter minimum illumination mass to filter detection")
particleTolerance = int(input())

print("Enter the frame to start from")
startFrame = int(input())

print("Enter the number of frames to evaluate")
frameLength = int(input())

t = evaluate_features(videoName, particleSize, particleTolerance, startFrame, frameLength)
t = evaluate_trajectories(t, 50, 10, 0.99, 3)
t = fixed_filter_stubs(t, 10)

particles = extract_particles(t)

t = postfiltering(t, particles, 5000, 3)
t = unfilter_jumps(t, particles, 3)

#print(t)

export(t, particles)

ax = tp.plot_traj(t)
plt.show()


