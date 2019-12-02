# Author: Nihesh Anderson
# Date	: Oct 23, 2019

import cv2
import numpy as np
from utils import homography_estimator, moving_average_filter
from Args import FPS, MOVING_AVERAGE_LENGTH, MAX_EPOCHS, CROP 
from homography import myhomography
import scipy.misc

class VideoStabiliser:

	def __init__(self, src):

		self.src = src

		self.capture = cv2.VideoCapture(self.src)
		self.status, self.prev_frame = self.capture.read()
		
		self.h = self.prev_frame.shape[0]
		self.w = self.prev_frame.shape[1]

		self.motion_data = []

	def learn_motion(self):

		it = 0

		while(self.status and it < MAX_EPOCHS):

			self.status, cur_frame = self.capture.read()

			if(not self.status):
				break
			
			homography = np.round_(myhomography(self.prev_frame, cur_frame))
			# homography = homography_estimator(self.prev_frame,cur_frame)
			self.motion_data.append(homography)

			self.prev_frame = cur_frame
			it += 1

		self.motion_data = np.asarray(self.motion_data)

		self.capture.release()
		cv2.destroyAllWindows()

	def smoothen(self):

		self.motion_data = moving_average_filter(self.motion_data, averaging_length = MOVING_AVERAGE_LENGTH)

	def saveVideo(self):

		n = self.motion_data.shape[0]
		capture = cv2.VideoCapture(self.src)
		# fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		out = cv2.VideoWriter('./output.avi', fourcc, FPS, (2560, 720))

		for i in range(n):

			status, frame = capture.read()

			if(not status):
				break

			output = cv2.warpAffine(frame, self.motion_data[i][:-1], (self.w, self.h))
			output = output[CROP:-CROP, CROP:-CROP]
			output = scipy.misc.imresize(output, (self.h, self.w), interp = "bilinear", mode = None)

			frame = np.concatenate([frame, output], axis = 1)

			out.write(frame)

		capture.release()
		out.release()
		cv2.destroyAllWindows()

if(__name__ == "__main__"):

	video_stabiliser = VideoStabiliser("samples/video.mp4")
	video_stabiliser.learn_motion()
	video_stabiliser.smoothen()
	video_stabiliser.saveVideo()

	pass