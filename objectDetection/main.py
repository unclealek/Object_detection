import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import imutils
from deepface import deepface


#read image
img = cv2.imread('shapes_and_colors.jpg')

#display
cv2.imshow('origial', img)
cv2.waitKey(0)

#convert to gryscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#display
cv2.imshow('gray image',img_gray)
cv2.waitKey(0)

#apply binary threshold
ret, img_thresh = cv2.threshold(img_gray,200,255, cv2.THRESH_BINARY)
#display
cv2.imshow('binary image', img_thresh)
cv2.waitKey(0)

#detect contour from binary image
contours, _ = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#print("Number of contours found = {}".format(len(contours)))

#draw contour on original image
img_copy = img.copy()
cv2.drawContours(img_copy, contours, -1, (0,255,0), 2, lineType=cv2.LINE_AA)
#display
cv2.imshow('contour image',img_copy)
cv2.waitKey(0)



cv2.destroyAllWindows()
print("Number of contours found = {}".format(len(contours)))
