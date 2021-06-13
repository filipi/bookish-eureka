#!/usr/bin/env python

#https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html

import numpy as np
import cv2
import sys

print ("opening video...")
#cap = cv2.VideoCapture('slow.flv')
#####################################################
#cap = cv2.VideoCapture('../obtainingData/basin_DNS.avi')
#cap = cv2.VideoCapture('obtaningData/Simpson/frontSimpson.mpg')

cap = cv2.VideoCapture('../obtaningData/Neufeld/neufeld.mpg')
#cap = cv2.VideoCapture('../obtaningData/LR_Re895.mp4')

#cap = cv2.VideoCapture('../obtaningData/lockExchangeSimpson_filipi_Re2445/test.mp4')
#cap = cv2.VideoCapture('../obtaningData/2_fast.mpg')


# params for ShiTomasi corner detection
print("setting parameters")
feature_params = dict( maxCorners = 100, #100,
                       qualityLevel = 0.3, #0.3,
                       minDistance = 7, #7,
                       blockSize = 7 ) #7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (50, 50), #(15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))

print(color[1].tolist())

# Take first frame and find corners in it
#for i in range(0, 25):
#    ret, old_frame = cap.read()
ret, old_frame = cap.read()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

#i = 0
#print(p0)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
mask = (255-mask)

while(1):
#for i in range(0, 76):
    p1 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)    
    for j,(p) in enumerate(p1):
        x,y = p.ravel()        
        mostra = cv2.circle(old_frame,(x,y), 4,[0, 0, 255],-1)
        #img = cv2.add(old_frame, mostra)
        mask = np.bitwise_and(mask, mostra)
        
        #print(x, y, color[j].tolist())
    cv2.imshow('frame', mask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    if k == 32:
        #i += 1
        #print(i)
        ret, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
quit()

