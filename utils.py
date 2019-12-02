	# Author: Nihesh Anderson
# Date	: Oct 23, 2019

import numpy as np
import cv2
import math
from Args import NUM_CORRESPONDENCES

def moving_average_filter(data, averaging_length = 3):


	data = np.copy(data)
	orig_data = np.copy(data)
	data = np.cumsum(data, axis = 0)
	summed_data = np.copy(data)
	for i in range(len(data)):
		data[i] = np.mean(data[i: i + averaging_length], axis = 0)

	difference = data - summed_data 
	transformed_data = difference + orig_data

	return transformed_data


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

	points_to_track = cv2.goodFeaturesToTrack(prev_frame, maxCorners = NUM_CORRESPONDENCES, qualityLevel = 0.01, minDistance = 30)
	assert(points_to_track is not None)

	# visualise_points(prev_frame, points_to_track)

	point_match, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, cur_frame, points_to_track, None)

	# visualise_points(cur_frame, point_match)

	# Pick only the points that are valid matches
	valid_correspondences = np.argwhere(status == 1)[:, 0]
	points_to_track = points_to_track[valid_correspondences]
	point_match = point_match[valid_correspondences]


	homography, _ = cv2.estimateAffinePartial2D(points_to_track, point_match)

	# homography = np.zeros([3, 3])
	# homography[0][0] = homography[1][1] = homography[2][2] = 1

	# visualise_homography(prev_frame, homography)

	return homography

# Function to display image
def display(image):
    cv2.imshow("image",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to pad image for convolution
def pad(image, kernelX, kernelY):
    imageX, imageY = image.shape    
    kernelXmid, kernelYmid = kernelX//2, kernelY//2
    image_padded = np.zeros((imageX+kernelX-1,imageY+kernelY-1),dtype=np.uint8)
    #padding the center
    image_padded[kernelXmid:imageX+kernelXmid,kernelYmid:imageY+kernelYmid] = image 
    #paddng top
    image_padded[0:kernelXmid, kernelYmid:imageY+kernelYmid] = image[0:kernelXmid,:]
    #padding bottom
    image_padded[imageX+kernelXmid:, kernelYmid:imageY+kernelYmid] = image[imageX-kernelXmid:,:]
    #padding left
    image_padded[:, 0:kernelYmid] = image_padded[:,kernelYmid:2*kernelYmid]
    #padding right
    image_padded[:, imageY+kernelYmid:] = image_padded[:,imageY:imageY+kernelYmid]
    return image_padded

# Function to perform convolution
def conv(image, kernel):
    kernelX, kernelY = kernel.shape
    kernelXmid, kernelYmid = kernelX//2, kernelY//2
    imageX, imageY = image.shape
    new_image = np.zeros(image.shape,dtype=np.uint8)
    image_padded = pad(image, kernelX, kernelY)
    for i in range(kernelXmid, imageX+kernelXmid):
        for j in range(kernelYmid,imageY+kernelYmid):
            new_image[i-kernelXmid][j-kernelYmid] = np.sum(image_padded[i-kernelXmid:i+kernelXmid+1, j-kernelYmid:j+kernelYmid+1]*kernel)
    return new_image

# Function to general a gaussian kernel
def gaussian(kernel, sigma):
    K = np.zeros((kernel, kernel))
    kernelX, kernelY = K.shape
    kernelXmid, kernelYmid = kernelX//2, kernelY//2
    weight = 0
    for i in range(kernelX):
        for j in range(kernelY):
            # K[i][j] = int(100*math.exp(-1*((i-kernelXmid)**2 + (j-kernelYmid)**2)/(2*(sigma**2))))
            K[i][j] = math.exp(-1*((i-kernelXmid)**2 + (j-kernelYmid)**2)/(2*(sigma**2)))
            weight+= K[i][j]
    return K, weight

def downsample(image, scale):
    return np.array(image[::scale,::scale])

def rotate90(image):
    rotated = np.zeros(image.shape[::-1], dtype=type(image[0,0]))
    for i in range(image.shape[1]):
        rotated[i,:] = image[:,i].T[::-1]
    return rotated



if(__name__ == "__main__"):

	pass