#!/usr/bin/python
#https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html
import numpy as np
import cv2
feature_params = dict( maxCorners = 100, #100, # params for ShiTomasi corner detection
                       qualityLevel = 0.2, #0.3,
                       minDistance = 12, #7,
                       blockSize = 7 ) #7 )
lk_params = dict( winSize  = (200, 200), #(15,15), # Parameters for lucas kanade optical flow
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
color = np.random.randint(0,255,(100,3)) # Create some random colors

#cap = cv2.VideoCapture('basin_DNS.avi')
#cap = cv2.VideoCapture('obtaningData/Simpson/frontSimpson.mpg')
cap = cv2.VideoCapture('obtaningData/Neufeld/neufeld.mpg')

#cap = cv2.VideoCapture('frenteSimpson.mpg')
pause = False
frameNumber = 0

ret, old_frame = cap.read()
mask = np.zeros_like(old_frame)
mask = (255-mask)
frame = old_frame

cv2.imshow('frame', frame)
while(1):
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    if k == 32:
        #print(frameNumber)
        frameNumber = frameNumber + 1
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY) 
        p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
        frame_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        #p1 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)        
        print(p1)
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        # draw the tracks
        for k,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            #print(a)
            #quit()
            mask = cv2.line(mask, (a,b),(c,d), color[k].tolist(), 5)
            frame = cv2.circle(old_frame,(a,b),5,color[k].tolist(),-1)
            img = cv2.add(frame, mask)
            #mask = mask + frame
            mask = np.bitwise_and(mask, frame)
            cv2.imshow('frame', mask)
            
            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1,1,2)
        ret, old_frame = cap.read()
        
quit()

#adquire
#
