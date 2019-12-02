# Author: Nihesh Anderson
# Date	: Oct 23, 2019

import cv2
import numpy as np
from utils import homography_estimator, moving_average_filter
from Args import FPS, MOVING_AVERAGE_LENGTH, MAX_EPOCHS, CROP, FILE 
from homography import myhomography
import scipy.misc
from matplotlib import pyplot as plt

DIM = None

class VideoStabiliser:

	def __init__(self, src):

		self.src = src

		self.capture = cv2.VideoCapture(self.src)
		self.status, self.prev_frame = self.capture.read()
		
		self.h = self.prev_frame.shape[0]
		self.w = self.prev_frame.shape[1]

		self.motion_data = []

	def get_params(self, homography):

		x_movement = homography[0][2]
		y_movement = homography[1][2]

		angle = np.arctan2(homography[1][0], homography[0][0])

		return [ x_movement, y_movement, angle ]

	def restoreParams(self, vector):

		homography = np.zeros((3, 3))
		homography[1][1] = homography[0][0] = np.cos(vector[2])
		homography[0][1] = -np.sin(vector[2])
		homography[1][0] = -homography[0][1]
		homography[0][2] = vector[0]
		homography[1][2] = vector[1]
		homography[2][2] = 1

		return homography

	def learn_motion(self):

		global DIM

		it = 0

		while(self.status and it < MAX_EPOCHS):

			self.status, cur_frame = self.capture.read()
			DIM = cur_frame.shape

			if(not self.status):
				break
			
			# homography = np.round_(myhomography(self.prev_frame, cur_frame))
			homography = homography_estimator(self.prev_frame,cur_frame)
			self.motion_data.append(self.get_params(homography))

			self.prev_frame = cur_frame
			it += 1

		self.motion_data = np.asarray(self.motion_data)

		self.capture.release()
		cv2.destroyAllWindows()

	def smoothen(self):

		self.before_smoothing = []

		for i in range(self.motion_data.shape[0]):
			self.before_smoothing.append(np.linalg.norm(self.motion_data[i]))

		plt.clf()
		plt.plot(self.before_smoothing)
		plt.title("Homography magnitude vs frame")
		plt.xlabel("Frame number")
		plt.ylabel("Magnitude")
		plt.savefig("./output/AffineCurveBeforeSmoothening.jpg")

		self.motion_data = moving_average_filter(self.motion_data, averaging_length = MOVING_AVERAGE_LENGTH)

		self.after_smoothing = []

		for i in range(self.motion_data.shape[0]):
			self.after_smoothing.append(np.linalg.norm(self.motion_data[i]))


		plt.clf()
		plt.plot(self.after_smoothing)
		plt.title("Homography magnitude vs frame")
		plt.xlabel("Frame number")
		plt.ylabel("Magnitude")
		plt.savefig("./output/AffineCurveAfterSmoothening.jpg")

	def saveVideo(self):

		global DIM

		n = self.motion_data.shape[0]
		capture = cv2.VideoCapture(self.src)
		# fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		out = cv2.VideoWriter('./output/' + FILE + ".avi", fourcc, FPS, (DIM[1] * 2, DIM[0]))

		for i in range(n):

			status, frame = capture.read()

			if(not status):
				break

			output = cv2.warpAffine(frame, self.restoreParams(self.motion_data[i])[:-1], (self.w, self.h))
			output = output[CROP:-CROP, CROP:-CROP]
			output = scipy.misc.imresize(output, (self.h, self.w), interp = "bilinear", mode = None)

			frame = np.concatenate([frame, output], axis = 1)

			out.write(frame)

		capture.release()
		out.release()
		cv2.destroyAllWindows()

if(__name__ == "__main__"):

	video_stabiliser = VideoStabiliser("./samples/" + FILE + ".mp4")
	video_stabiliser.learn_motion()
	video_stabiliser.smoothen()
	video_stabiliser.saveVideo()

	pass