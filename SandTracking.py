#PlotPy for Falling Sand

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import trackpy as tp

import av
import pims
import math

import pandas as pd
from pandas import DataFrame, Series

frames_per_second = 1000
"""The frames per second of the camera being used"""

class particle:
	"""This is a class representing a single particle found by Trackpy"""

	used_index = []
	"""The integer list of DataFrame indices generated by Trackpy that pass filtering."""

	def __init__(self, ID):
		self.index = []
		"""The integer list of DataFrame indicies generated by Trackpy that the particle uses"""
		self.ID = ID
		"""The integer particle ID assigned by Trackpy when linking trajectories from features."""
		self.diameter = 0
		"""The double averaged size of the particle in the frames it appears in, in pixels."""
		self.ecc = 0
		"""The double averaged eccentricity of the particle in the frames it appears in."""
		self.polyline = None
		"""The Numpy Polyline representing a quadratic polynomial line of best fit for the particle's position."""
		self.pos_derivative = ([], [], [], [])		
		"""The tuple of the 0th, 1st, 2nd and 3rd derivatives of position, which is encoded as a list of tuples in the form
		of (X, Y, Error, Frame)"""
		self.irregular = []
		"""The integer list of indicies where an irregular motion occurs."""
		self.average = []
		"""he list of tuples representing a particle's average derivatives of position using the tuple (X, Y)."""

	def add_index(self, coord, diameter, ecc, index):
		"""
		The method that adds the information in an index of TrackPy's trajectory DataFrame to it's associated particle.
			
		Parameters:
			coord (single tuple of objects): The positional information of the particle using tuple (x (double), y (double), error (double), frame(double)).
			diameter (double): The diameter of the particle in pixels.
			ecc (double): The eccentricty of the particle.
			index (int): the index of the DataFrame being evaluated.
		"""
		self.diameter = (self.diameter * len(self.pos_derivative[0]) + diameter) / (len(self.pos_derivative[0]) + 1)
		self.ecc = (self.ecc * len(self.pos_derivative[0]) + ecc) / (len(self.pos_derivative[0]) + 1)
		self.index.append(index)
		self.pos_derivative[0].append(coord)

		if index not in self.used_index:
			self.used_index.append(index)

	def calc_polyline(self):
		"""
			The method that calculates a quadratic line of best fit for the particle's position coordinates.
		"""
		x = []
		y = []

		for i in range(0, len(self.pos_derivative[0])):
			x.append(self.pos_derivative[0][i][0])
			y.append(self.pos_derivative[0][i][1])

		self.polyline = np.poly1d(np.polyfit(np.array(x), np.array(y), 2))

	def calc_vel(self, angle, x_restriction = 0, y_restriction = 0):
		"""
			The method that calculates the particle's velocity from its position.

			Parameters:
				angle (double): The maximum angle in radians a particle can move between positions before it's considered an irregular motion.
				x_restriction (double): The tuple describing the restriction of x motion, all motion in the sign of x_restriction is considered irregular.
				y_restriction (double): The tuple describing the restriction of y motion, all motion in the sign of x_restriction is considered irregular.
		"""
		self.average.append((0, 0))

		for i in range (0, len(self.pos_derivative[0]) - 1):
			dt = (self.pos_derivative[0][i + 1][3] - self.pos_derivative[0][i][3]) / frames_per_second
			t = (self.pos_derivative[0][i + 1][3] + self.pos_derivative[0][i][3]) / 2

			err = ((self.pos_derivative[0][i][2] * self.pos_derivative[0][i + 1][0] - self.pos_derivative[0][i + 1][2] * self.pos_derivative[0][i][0]) / dt, 
				(self.pos_derivative[0][i][2] * self.pos_derivative[0][i + 1][1] - self.pos_derivative[0][i + 1][2] * self.pos_derivative[0][i][1]) / dt)

			self.pos_derivative[1].append(((self.pos_derivative[0][i + 1][0] - self.pos_derivative[0][i][0]) / dt, 
				(self.pos_derivative[0][i + 1][1] - self.pos_derivative[0][i][1]) / dt, err, t))
			self.average[-1]  = (self.average[-1][0] + self.pos_derivative[1][-1][0], self.average[-1][1] + self.pos_derivative[1][-1][1])

			if len(self.pos_derivative[1]) > 1 and (get_cos(self.pos_derivative[1][-1], self.pos_derivative[1][-2]) < math.cos(angle) or 
				self.pos_derivative[1][-1][0] * x_restriction > 0 or self.pos_derivative[1][-1][1] * y_restriction > 0):
				self.irregular.append(self.index[i])

		self.average[-1] = (self.average[-1][0] / len(self.pos_derivative[1]), self.average[-1][1] / len(self.pos_derivative[1]))

	def calc_accel(self):
		"""The method that calculates a particle's acceleration from its velocity."""
		self.average.append((0, 0))

		for i in range (0, len(self.pos_derivative[1]) - 1):
			dt = (self.pos_derivative[1][i + 1][3] - self.pos_derivative[1][i][3]) / frames_per_second
			t = (self.pos_derivative[1][i + 1][3] + self.pos_derivative[1][i][3]) / 2

			err = ((self.pos_derivative[1][i][2][0] * self.pos_derivative[1][i + 1][0] - self.pos_derivative[1][i + 1][2][0] * self.pos_derivative[1][i][0]) / dt, 
				(self.pos_derivative[1][i][2][1] * self.pos_derivative[1][i + 1][1] - self.pos_derivative[1][i + 1][2][1] * self.pos_derivative[1][i][1]) / dt)

			self.pos_derivative[2].append(((self.pos_derivative[1][i + 1][0] - self.pos_derivative[1][i][0]) / dt, 
				(self.pos_derivative[1][i + 1][1] - self.pos_derivative[1][i][1]) / dt, err, t))
			self.average[-1]  = ((self.average[-1][0] * len(self.pos_derivative[2]) + self.pos_derivative[2][-1][0]) / (len(self.pos_derivative[2]) + 1), 
								(self.average[-1][1] * len(self.pos_derivative[2]) + self.pos_derivative[2][-1][1]) / (len(self.pos_derivative[2]) + 1))

	def calc_jerk(self):
		"""The method that calculates a particle's jerk from its acceration."""
		self.average.append((0, 0))

		for i in range (0, len(self.pos_derivative[2]) - 1):
			dt = (self.pos_derivative[2][i + 1][3] - self.pos_derivative[2][i][3]) / frames_per_second
			t = (self.pos_derivative[2][i + 1][3] + self.pos_derivative[2][i][3]) / 2

			err = ((self.pos_derivative[2][i][2][0] * self.pos_derivative[2][i + 1][0] - self.pos_derivative[2][i + 1][2][0] * self.pos_derivative[2][i][0]) / dt, 
				(self.pos_derivative[2][i][2][1] * self.pos_derivative[2][i + 1][1] - self.pos_derivative[2][i + 1][2][1] * self.pos_derivative[2][i][1]) / dt)

			self.pos_derivative[3].append(((self.pos_derivative[2][i + 1][0] - self.pos_derivative[2][i][0]) / dt, 
				(self.pos_derivative[2][i + 1][1] - self.pos_derivative[2][i][1]) / dt, err, t))
			self.average[-1]  = (self.average[-1][0] + self.pos_derivative[3][-1][0], self.average[-1][1] + self.pos_derivative[3][-1][1])

	def analyze(self, angle, x_restriction = 0, y_restriction = 0):
		"""
			The method that calculates all derivatives of position and the polyline of a particle.

			Parameter:
				angle (double): The maximum angle in radians a particle can move between positions before it's considered an irregular motion.
				x_restriction (double): The tuple describing the restriction of x motion, all motion in the sign of x_restriction is considered irregular.
				y_restriction (double): The tuple describing the restriction of y motion, all motion in the sign of x_restriction is considered irregular.
		"""
		self.calc_polyline()
		self.calc_vel(angle, x_restriction, y_restriction)
		self.calc_accel()
		self.calc_jerk()

def get_cos(v1, v2):
	#Helper function to get the angle of two vectors.

	numerator = v1[0] * v2[0] + v1[1] * v2[1]
	denominator = math.sqrt(math.pow(v1[0], 2) + math.pow(v1[1], 2)) * math.sqrt(math.pow(v2[0], 2) + math.pow(v2[1], 2))

	return numerator / denominator

@pims.pipeline
def to_grey(frame):
	"""
		The function that converts a NumpyArray to greyscale.

		Parameters:
			frame (NumpyArray): The NumpyArray representing a video frame.

		Returns:
			frame (NumpyArray): The NumpyArray taken in and converted to greyscale.
	"""
	red = frame[:, :, 0]
	blue = frame[:, :, 1]
	green = frame[:, :, 2]

	return red * 0.2125 + green * 0.7154 + blue * 0.0721

def process_video(video_name):
	"""
		The function that converts a video to a list of greyscale NumpyArrays.

		Parameters:
			video_name (String): The name of the video to be evaluated stored in the Recordings folder.
	"""
	return to_grey(pims.PyAVReaderTimed(video_name))

def evaluate_features(video_name, particle_size, particle_minmass, start_frame, length, noise):
	"""
		The function that  runs Trackpy's feature detection algorithm on the arrays.

		Parameters:
			video_name (String): The name of the video to be evaluated stored in the Recordings folder.
			particle_size (int): The odd-number size of the feature to be detected by Trackpy.
			particle_minmass (double): The minimum feature brightness to filter using Trackpy's filtering functions.
			start_frame (int): The frame in the video from which to begin evaluation.
			length (int): The number of frames to evaluate.

		Returns:
			frame (DataFrame): The DataFrame of the features found in the video.
	"""
	video_frames = process_video(video_name)
	f = tp.batch(video_frames[start_frame: start_frame + length], particle_size, minmass = particle_minmass, noise_size = noise)

	return f

def evaluate_trajectories(data_frame, search_size, lb_search_size, step, particle_memory):
	"""
		The function that links features previously discovered by Trackpy and creates a trajectory.

		Parameters:
			data_frame (DataFrame): The DataFrame of feature inforation generated by Trackpy.
			search_size (int): The radius of pixels the trajectory searching program will look for the particle.
			lb_search_size (int): The lower bound of the search size.
			step (double): The rate at which the search size decreases to the lb_search_size.
			particle_memory (int): The number of frames that a particle cannot be found before it is pruned from memory.

		Returns:
			t (DataFrame): The DataFrame of trajectory information generated from the DataFrame of features.
	"""
	pred = tp.predict.NearestVelocityPredict()				
	t = pred.link_df(data_frame.copy(), search_size, adaptive_stop = lb_search_size, adaptive_step = step, memory = particle_memory)

	return t

def extract_particles(data_frame):
	"""
		The function that generates particles with all relavent information from a DataFrame of trajectory.

		Parameters:
			data_frame (DataFrame): The DataFrame of trajectory information generated by Trackpy.

		Returns:
			particles (Dictionary of particles): The Dictionary containing all the particles extracted from the input DataFrame with the format {keys = ID: values: particle}.
	"""
	particles = dict()

	for i in range (0, data_frame.shape[0]):
		ID = data_frame.at[i, "particle"]

		if ID not in particles:
			particles[ID] = particle(ID)

		particles[ID].add_index((data_frame.at[i, "x"], data_frame.at[i, "y"], data_frame.at[i, "ep"] / 2, data_frame.at[i, "frame"]), 
			data_frame.at[i, "size"] * 2, data_frame.at[i, "ecc"], i)

	return particles

def split(data_frame, p, filter_stub, angle, x_restriction = 0, y_restriction = 0):
		"""
			The function that splits a single trajectory into recoverable trajectories if there is an irregular motion in the original trajectory.

			Parameters:
				data_frame (DataFrame): The DataFrame of trajectory information generated by Trackpy.
				p (particle): The particle that displayed irregular motion.
				filter_stub (int): The minimum amount of frames the recoverable trajectories must persist for.
				angle (double): The maximum angle in radians a particle can move between positions before it's considered an irregular motion.
				x_restriction (double): The tuple describing the restriction of x motion, all motion in the sign of x_restriction is considered irregular.
				y_restriction (double): The tuple describing the restriction of y motion, all motion in the sign of x_restriction is considered irregular.

			Returns:
				return_particles (list of particles): The list of particles generated from the input irregular particle.
		"""
		return_particles = []

		for x in range(0, len(p.irregular) + 1):
			return_particles.append(particle(data_frame.max()[9] + 1 + x))

		return_index = 0

		for x in p.index:
			if x in p.irregular:
				return_index += 1

			return_particles[return_index].add_index((data_frame.at[x, "x"], data_frame.at[x, "y"], data_frame.at[x, "ep"] / 2, data_frame.at[x, "frame"]), 
													data_frame.at[x, "size"] * 2, data_frame.at[x, "ecc"], x)

		for rp in return_particles.copy():
			if len(rp.pos_derivative[0]) < filter_stub:
				for x in range(0, len(return_particles)):
					if return_particles[x].ID == rp.ID:
						return_particles.pop(x)
						break;

				for x in rp.index:
					particle.used_index.remove(x)

		for rp in return_particles:
			rp.analyze(angle, x_restriction, y_restriction)

			for x in rp.index:
				data_frame.at[x, "particle"] = rp.ID

		data_frame = data_frame.loc[particle.used_index]

		return return_particles

def merge(data_frame, particles, error_tolerance, angle, x_restriction = 0, y_restriction = 0):
	"""
		The function that takes a list of previously split particles and merges them if they're very similar in trajectory.

		Parameters:
			data_frame (DataFrame): The DataFrame of trajectory information generated by Trackpy.
			particles (list of particles): The list of particle generated from an irregular particle.
			error_tolerance (double): The maximum sum of residuals between a particle's polyline and another particle's position coordinates allowing merging.
			angle (double): The maximum angle in radians a particle can move between positions before it's considered an irregular motion.
			x_restriction (double): The tuple describing the restriction of x motion, all motion in the sign of x_restriction is considered irregular.
			y_restriction (double): The tuple describing the restriction of y motion, all motion in the sign of x_restriction is considered irregular.

		Returns:
			particles (list of particles): The list of particles after remerging previously split particles.
	"""
	evaluated_particles = []

	for p1 in particles.copy():
		evaluated_particles.append(p1.ID)
		poly = p1.polyline
		lse = 0

		for p2 in particles.copy():
			if p2.ID not in evaluated_particles:
				for i in range(0, len(p2.pos_derivative[0])):
					lse += math.pow(poly(p2.pos_derivative[0][i][0]) - p2.pos_derivative[0][i][1], 2)

				lse = math.sqrt(lse / len(p2.pos_derivative[0]))

				if lse < error_tolerance:
					evaluated_particles.append(p2.ID)

					for i in range(0, len(particles)):
						if particles[i].ID == p1.ID:
							for x in p2.index:
								data_frame.at[x, "particle"] = particles[i].ID
								particles[i].add_index((data_frame.at[x, "x"], data_frame.at[x, "y"], data_frame.at[x, "ep"], data_frame.at[x, "frame"]), 
													data_frame.at[x, "size"] * 2, data_frame.at[x, "ecc"], x)

							particles[i].analyze(angle, x_restriction, y_restriction)
							break

						if particles[i].ID == p2.ID:
							particles.pop(i)
							break

	return particles

def unfilter_jumps(data_frame, particles, filter_stub, error_tolerance, angle, x_restriction = 0, y_restriction = 0):
	"""
		The function that fixes irregular trajectories detected by Trackpy.
		
		Parameters:
			data_frame (DataFrame): The DataFrame of trajectory information generated by Trackpy.
			particles (Dictionary of particles): The Dictionary containing all the particles extracted from the trajectory DataFrame with the format {keys = ID: values: particle}.
			filter_stub (int): The minimum amount of frames the recoverable trajectories must persist for.
			error_tolerance (double): The maximum sum of residuals between a particle's polyline and another particle's position coordinates allowing merging.
			angle (double): The maximum angle in radians a particle can move between positions before it's considered an irregular motion.
			x_restriction (double): The tuple describing the restriction of x motion, all motion in the sign of x_restriction is considered irregular.
			y_restriction (double): The tuple describing the restriction of y motion, all motion in the sign of x_restriction is considered irregular.

		Returns:
			data_frame (DataFrame): The DataFrame of trajectory information with all unrecoverable particle trajectories removed.
	"""
	for p in particles.copy().values():
		if not len(p.irregular) == 0:
			new_particles = merge(data_frame, split(data_frame, p, filter_stub, angle), error_tolerance, angle)
			particles.pop(p.ID)

			for split_particles in new_particles:
				particles[split_particles.ID] = split_particles

	return data_frame.loc[particle.used_index]

def fixed_filter_stubs(data_frame, i):
	"""
		The function that fixes Trackpy's filter_stubs function by restoring the indices of the data_frame.

		Parameters:
			data_frame (DataFrame): The DataFrame of trajectory information generated by Trackpy.
			i (int): The minimum amount of frames the recoverable trajectories must persist for.

		Returns:
			t (Data_Frame): The DataFrame of the trajectory information of particles that exist in more frames than i.
	"""
	t = tp.filter_stubs(data_frame.copy(), i)
	t.index = range(0, len(t))

	return t

def postfiltering(data_frame, particles, stillness, tolerance, angle, error_tolerance, filter_stub, x_restriction = 0, y_restriction = 0):
	"""
		The function that filters out particles based on its velocity, degree of irregularity, and how many frames it's in.

		Parameters:
			data_frame (DataFrame): The DataFrame of trajectory information generated by Trackpy.
			particles (Dictionary of particles): The Dictionary containing all the particles extracted from the trajectory DataFrame with the format {keys = ID: values: particle}.
			stillness (double): The minimum speed of a particle in pixels before it is considered a still object being misdetected by Trackpy.
			tolerance (int): The maximum amount of irregularities a particle can have before it is considered too irregular to evaluate.
			error_tolerance (double): The maximum sum of residuals between a particle's polyline and another particle's position coordinates allowing merging.
			filter_stub (int): The minimum amount of frames the recoverable trajectories must persist for.
			x_restriction (double): The tuple describing the restriction of x motion, all motion in the sign of x_restriction is considered irregular.
			y_restriction (double): The tuple describing the restriction of y motion, all motion in the sign of x_restriction is considered irregular.

		Returns:
			data_frame (DataFrame): The DataFrame of trajectory information filtered by the function.

	"""
	if tolerance == None:
		tolerance = math.inf

	for p in particles.copy().values():
		p.analyze(angle, x_restriction, y_restriction)

		if (abs(p.average[0][1]) < stillness and abs(p.average[0][1] < stillness)) or len(p.irregular) > tolerance:
			particles.pop(p.ID)

			for i in p.index:
				particle.used_index.remove(i)

	unfilter_jumps(data_frame, particles, filter_stub, error_tolerance, angle, x_restriction, y_restriction)

	return data_frame.loc[particle.used_index]

def export(data_frame, particles = None):
	"""
		The function that exports all information to an excel sheet in the directory folder named "output.xlsx"

		Parameters:
			data_frame (DataFrame): The DataFrame of trajectory information generated by Trackpy.
			particles (Dictionary of particles): The Dictionary containing all the particles extracted from the input DataFrame with the format {keys = ID: values: particle}.
	"""
	if particles == None:
		raw_data = data_frame.copy()
		raw_data.to_excel("output.xlsx", sheet_name = "raw_data")

	else:
		raw_data = data_frame.copy()
		velocity_data = dict({"x_vel": [], "y_vel": [], "x_err": [], "y_err": [], "frame": [], "particle": []})
		acceleration_data = dict({"x_accel": [], "y_accel": [], "x_err": [], "y_err": [], "frame": [], "particle": []})
		jerk_data = dict({"x_jerk": [], "y_jerk": [], "x_err": [], "y_err": [], "frame": [], "particle": []})

		for p in particles.values():
			for i in range (0, len(p.pos_derivative[1])):
				velocity_data["x_vel"].append(p.pos_derivative[1][i][0])
				velocity_data["y_vel"].append(p.pos_derivative[1][i][1])
				velocity_data["x_err"].append(p.pos_derivative[1][i][2][0])
				velocity_data["y_err"].append(p.pos_derivative[1][i][2][1])
				velocity_data["frame"].append(p.pos_derivative[1][i][3])
				velocity_data["particle"].append(p.ID)

				if i < len(p.pos_derivative[2]):
					acceleration_data["x_accel"].append(p.pos_derivative[2][i][0])
					acceleration_data["y_accel"].append(p.pos_derivative[2][i][1])
					acceleration_data["x_err"].append(p.pos_derivative[2][i][2][0])
					acceleration_data["y_err"].append(p.pos_derivative[2][i][2][1])
					acceleration_data["frame"].append(p.pos_derivative[2][i][3])
					acceleration_data["particle"].append(p.ID)

				if i < len(p.pos_derivative[3]):
					jerk_data["x_jerk"].append(p.pos_derivative[3][i][0])
					jerk_data["y_jerk"].append(p.pos_derivative[3][i][1])
					jerk_data["x_err"].append(p.pos_derivative[3][i][2][0])
					jerk_data["y_err"].append(p.pos_derivative[3][i][2][1])
					jerk_data["frame"].append(p.pos_derivative[3][i][3])
					jerk_data["particle"].append(p.ID)

		velocity_data = pd.DataFrame.from_dict(velocity_data)
		acceleration_data = pd.DataFrame.from_dict(acceleration_data)
		jerk_data = pd.DataFrame.from_dict(jerk_data)

		with pd.ExcelWriter("output.xlsx") as writer:
			raw_data.to_excel(writer, sheet_name = "raw data")
			velocity_data.to_excel(writer, sheet_name = "velocity data")
			acceleration_data.to_excel(writer, sheet_name = "acceleration data")
			jerk_data.to_excel(writer, sheet_name = "jerk data")

def hist(data_frame, bins, column):
	"""
		The function that puts all the information of a specified column of a DataFrame into a histogram.

		Parameters:
			data_frame (DataFrame): The DataFrame that is being evaluated.
			bins (int): The amount of bins that the information is put into.
			column (String): The name of the column to be plotted.
	"""
	plt.figure()
	fig, ax = plt.subplots()
	ax.hist(data_frame[column], bins = bins)
	ax.set(xlabel = column, ylabel = 'count')
	plt.show()

def traj_plot(data_frame):
	"""
		The function that graphs trajectory information.

		Parameters:
			data_frame (DataFrame): The DataFrame of trajectory information generated by Trackpy.
	"""
	plt.figure()
	tp.plot_traj(data_frame)
	plt.show()

def get_frame(video_name, i, particle_size = None):
	"""
		The function that displays a frame of the video, optionally with its features circled.

		Parameters:
			video_name (String): The name of the video file in the Recordings folder to be processed
			i (int): The frame of the video to display.
			particle_size (int): The odd number average pixel size of a feature. Leave blank in order to only display the frame.
	"""
	video_frames = process_video(video_name)
	fig = plt.figure()

	if particle_size is not None:	
		f = tp.locate(video_frames[i], particle_size)
		tp.annotate(f, video_frames[i])
		plt.show()
	else:
		plt.imshow(video_frames[i])










