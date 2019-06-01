# -*- coding: utf-8 -*-
"""
Created on Fri May 31 10:21:53 2019

@author: SEBASTIAN LAVERDE
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

mars01 = cv2.imread("hirise-map-proj-v3/map-proj-v3/ESP_011283_2265_RED-0013.jpg", 1)
mars01_gray = cv2.cvtColor(mars01, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(mars01, 100, 70)
cv2.imshow("Original", mars01)

cv2.imshow("Canny",edges)
res,thresh = cv2.threshold(mars01[:,:,0], 50, 255, cv2.THRESH_BINARY)
res2,thresh2 = cv2.threshold(mars01[:,:,0], 70, 255, cv2.THRESH_BINARY)
res3,thresh3 = cv2.threshold(mars01[:,:,0], 30, 255, cv2.THRESH_BINARY)
print(mars01.shape)
thresh_adapt = cv2.adaptiveThreshold(mars01_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)    #adaptive treshold
#_, contours, hierachy = cv2.findContours(thresh_adapt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #find countours in the thresh (binary img)
_, contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.imshow("thresh_50", thresh)
cv2.imshow("thresh_70", thresh2)
cv2.imshow("thresh_30", thresh3)
cv2.imshow("thresh_adapt", thresh_adapt)

filtered = []
for c in contours:
	if (cv2.contourArea(c) < 50 or cv2.contourArea(c) > 500):continue #this is a parameter to be set
	filtered.append(c)

print(len(contours))
print(len(filtered))

Moments_cx = []
Moments_cy = []

objects = np.zeros([mars01_gray.shape[0],mars01_gray.shape[1],3], 'uint8')
for c in filtered:
    col = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
    cv2.drawContours(objects,[c], -1, col, -1)
    area = cv2.contourArea(c)
    p = cv2.arcLength(c,True)
    
    M = cv2.moments(c)
    cx = int( M['m10']/M['m00']) #to calculate the centroid of an image we use the moments of an image
    cy = int( M['m01']/M['m00'])
    Moments_cx.append(cx)
    Moments_cy.append(cy)
    print("area: ",area,"perimeter: ",p)

mars01copy = mars01.copy()

i = 0
print(Moments_cx, ", ", Moments_cy)
for obj in range(len(Moments_cx)):
    cv2.circle(mars01copy, (Moments_cx[i],Moments_cy[i]), 10, (0,0,255), 1)
    i+=1

cv2.imshow("Original_method1", mars01copy)    
cv2.namedWindow("Contours",cv2.WINDOW_NORMAL)
cv2.imshow("Contours",objects)

_, contours2, hierachy2 = cv2.findContours(thresh3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

filtered2 = []
for c in contours2:
	if (cv2.contourArea(c) < 10 or cv2.contourArea(c) > 500):continue #this is a parameter to be set
	filtered2.append(c)

print("second length: ", len(contours2))
print("filtered: ", len(filtered2))

objects2 = np.zeros([mars01_gray.shape[0],mars01_gray.shape[1],3], 'uint8')

print("\nSecond test with threshold value in 30\n")

Moments_cx1 = []
Moments_cy1 = []

for c in filtered2:
    col = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
    cv2.drawContours(objects2,[c], -1, col, -1)
    area = cv2.contourArea(c)
    p = cv2.arcLength(c,True)
    
    M = cv2.moments(c)
    cx = int( M['m10']/M['m00']) #to calculate the centroid of an image we use the moments of an image
    cy = int( M['m01']/M['m00'])
    Moments_cx1.append(cx)
    Moments_cy1.append(cy)
    
    print("area: ",area,"perimeter: ",p)

i = 0
print(Moments_cx1, ", ", Moments_cy1)
for obj in range(len(Moments_cx1)):
    cv2.circle(mars01, (Moments_cx1[i],Moments_cy1[i]), 10, (0,255,0), 1)
    i+=1

cv2.imshow("Original_method2", mars01)
cv2.imshow("Gray", mars01_gray)
cv2.namedWindow("Contours_2",cv2.WINDOW_NORMAL)
cv2.imshow("Contours_2",objects2)

cv2.waitKey(0)
cv2.destroyAllWindows()
