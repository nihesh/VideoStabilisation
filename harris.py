# Reference: https://en.wikipedia.org/wiki/Harris_Corner_Detector
import numpy as np
import cv2
from utils import *
np.warnings.filterwarnings('ignore')

# Function to compute gradient in y direction
def fy(img):
    image = np.array(img)
    image[:-1,:] -= image[1:,:] 
    return image

# Function to compute gradient in x direction
def fx(img):
    image = np.array(img)
    image[:,:-1] -= image[:,1:]
    return image

# Function to compute structure tensor C
def generate_tensor(img, fx, fy, m, threshold):
    fxy = fx * fy
    # Count of non-corner pixels
    count = 0
    corners = np.zeros((img.shape))
    C = np.zeros((2,2))
    for i in range(img.shape[0]-m):
        for j in range(img.shape[1]-m):
            C[0,0] = np.linalg.norm(fx[i:i+m, j:j+m])
            C[1,1] = np.linalg.norm(fy[i:i+m, j:j+m])
            C[0,1] = C[1,0] = np.sum(fxy[i:i+m, j:j+m])
            eigen_values = np.linalg.eig(C)[0]
            # Check if all eigen values of structure tensor are greater than threshold
            if np.all(np.log10(eigen_values) > threshold):
                corners[i + m//2, j + m//2] += 1
            else:
                count += 1
    print(count)
    return corners

# Function to perform non max supression
def nonmax(corners, m):
    points = []
    sup = np.zeros(corners.shape)
    for i in range(corners.shape[0]-m):
        for j in range(corners.shape[1]-m):
            if np.sum(corners[i:i+m, j:j+m]) > (m**2)/2:
                sup[i+m//2, j+m//2] = 1
                points.append((i+m//2, j+m//2));
    return sup, points

def paint(Img, corners):
    I = np.array(Img)
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            if(corners[i,j] == 1):
                I[i,j,:] = [182, 89, 155]
    return I

# Driver function
def harris_features(img):
    # Apply gaussian filter to remove noisy artifacts
    kernel_size, sigma = 5, 1
    k = gaussian(kernel_size, sigma)
    img = conv(img, k[0]/k[1])
    # Scale down all values from 0 to 1
    img = img/255
    # Window size for structure tensor
    window_size = 3
    threshold = [-0.75]
    c = generate_tensor(img, fx(img), fy(img), window_size, threshold)
    non_max_window = 7
    c, pts = nonmax(c, non_max_window)
    orb = cv2.ORB_create()
    keypoints = [cv2.KeyPoint(i[1], i[0], 1) for i in pts]
    kp, des = orb.compute(img, keypoints)
    # I_t = paint(img2, c)
    # cv2.imwrite("img_corners.png", I_t)
    return kp, des


if __name__ == "__main__":
    # img1 = cv2.imread("./img_2.png", 0)
    # img2 = cv2.imread("./img_2.png")
    # kp = orb.detect(img1, None)
    # print(len(kp), type(kp))
    # orb.compute(img, kp)
    # t = harris_features(img1)
    # print(t)
    pass