# -*- coding: utf-8 -*-
"""Homography.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hwMnJnfwZm369QTTx00sv6GhPQcmsGwa
"""

#!pip install opencv-python==3.4.2.16
#!pip install opencv-contrib-python==3.4.2.16
import cv2
import numpy as np
import random
import harris


def myhomography(img1,img2):
  threshold = 0.6
  num_epochs = 1000
  img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
  img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
  # sift = cv2.xfeatures2d.SIFT_create()
  kp1, desc1 = harris.harris_features(img1, None)
  kp2, desc2 = harris.harris_features(img2, None)
  matches = matching(img1,img2,kp1,kp2,desc1,desc2) #to be done from scratch
  corr = []
  for match in matches:
    x1,y1 = kp1[match.queryIdx].pt
    x2,y2 = kp2[match.trainIdx].pt
    corr.append([[x1,y1],[x2,y2]])
  corr = np.array(corr)
  #print(corr.shape)
  max_inliers = 0
  best_H = np.zeros((3,3))
  best_H[0][0] = best_H[1][1] = best_H[2][2] = 1
  for i in range(num_epochs):
    iter_points = []
    for j in range(4):
      curr = corr[random.randrange(0,corr.shape[0])]
      iter_points.append(curr)
    A = []
    for point in iter_points:
      p1 = point[0]
      p2 = point[1]
      temp1 = [-p1[0],-p1[1],-1,0,0,0,p1[0]*p2[0],p1[1]*p2[0],p2[0]]
      temp2 = [0,0,0,-p1[0],-p1[1],-1,p1[0]*p2[1],p1[1]*p2[1],p2[1]]
      A.append(temp1)
      A.append(temp2)
    A = np.array(A)
    U,S,V = np.linalg.svd(A)
    H = np.reshape(V[8],(3,3))
    H = H/H[2][2]
    inliers = 0
    for j in range(corr.shape[0]):
      point1,point2 = corr[j]
      curr = np.array([point1[0],point1[1],1])
      pred = np.matmul(H,curr.T)
      pred = np.reshape(pred,(3))
      pred = pred/pred[2]
      real = np.array([point2[0],point2[1],1])
      dist = np.linalg.norm(pred-real)
      if(dist<5):
        inliers+=1
    if(inliers > max_inliers):
      max_inliers = inliers
      best_H = H
    if(max_inliers > threshold*corr.shape[0]):
      break
  return best_H

def matching(img1,img2,kp1,kp2,desc1,desc2):
  matcher = cv2.BFMatcher(cv2.NORM_L2, True)
  matches = matcher.match(desc1, desc2)
  return matches
'''
img1 = cv2.imread('image002.jpg',0) #test images
img2 = cv2.imread('image004.jpg',0)
H = homography(img1,img2)

w = img1.shape[0]
h = img1.shape[1]
target = cv2.warpPerspective(img1, H, (h, w))
cv2.imwrite("trans.png", target)
'''