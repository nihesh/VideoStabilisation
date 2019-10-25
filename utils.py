	# Author: Nihesh Anderson
# Date	: Oct 23, 2019

import numpy as np
import cv2
from Args import NUM_CORRESPONDENCES

def moving_average_filter(data, averaging_length = 3):

	for i in range(data.shape[0]):

		data[i] = np.mean(data[i : i + averaging_length], axis = 0)

	return data


def visualise_points(img, points):

	"""
	Just a temporary function for debugging and visualisation purposes
	"""

	points = points.reshape(NUM_CORRESPONDENCES, 2).astype(int)

	for i in range(-5, 6):
		for j in range(-5, 6):
			img[points[:, 1] + i, points[:, 0] + j] = 255

	cv2.imshow("Visualisation", img)
	cv2.waitKey(5000)
	
	exit(0)

def visualise_homography(img, homography):

	w = img.shape[0]
	h = img.shape[1]

	target = cv2.warpPerspective(img, homography, (h, w))

	cv2.imshow("Homography Visualisation", target)
	cv2.waitKey(5000)

	exit(0)

def homography_estimator(prev_frame, cur_frame):

	prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
	cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)

	points_to_track = cv2.goodFeaturesToTrack(prev_frame, maxCorners = NUM_CORRESPONDENCES, qualityLevel = 0.04, minDistance = 1)
	assert(points_to_track is not None)

	# visualise_points(prev_frame, points_to_track)

	point_match, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, cur_frame, points_to_track, None)

	# visualise_points(cur_frame, point_match)

	# Pick only the points that are valid matches
	valid_correspondences = np.argwhere(status == 1)[:, 0]
	points_to_track = points_to_track[valid_correspondences]
	point_match = point_match[valid_correspondences]

	try:
		homography, _ = cv2.getAffineTransform(points_to_track, point_match)
	except:
		homography = np.zeros([3, 3])
		homography[0][0] = homography[1][1] = homography[2][2] = 1

	# visualise_homography(prev_frame, homography)

	return homography

if(__name__ == "__main__"):

	pass