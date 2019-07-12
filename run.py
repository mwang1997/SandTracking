import SandTracking as st
import math

##Execution##

#Input Questions
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

t = st.evaluate_features(videoName, particleSize, particleTolerance, startFrame, frameLength, 4)
t = st.evaluate_trajectories(t, 50, 10, 0.99, 3)
t = st.fixed_filter_stubs(t, 10)

particles = st.extract_particles(t)

t = st.postfiltering(t, particles, 5000, math.inf, math.pi * 15 / 180, 10000, 10, y_restriction = -1)

st.traj_plot(t)
st.export(t, particles)



