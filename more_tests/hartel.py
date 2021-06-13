#!/usr/bin/env python
#https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html

import numpy as np
import cv2
import sys

print ("opening video...")
cap = cv2.VideoCapture('../obtaningData/Neufeld/neufeld.mpg')
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
# Take first frame and find corners in it
for i in range(0, 18):
    ret, old_frame = cap.read()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)


# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

ret,frame = cap.read()

pause = False

i = 0
progressArray = ['-', '\\', '|', '/' ]

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('channels.avi',fourcc, 20.0, (int(cap.get(3)),int(cap.get(4))))

while(1):    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    if k == 32:
        pause = not pause
    if not pause:
        ret,frame = cap.read()
        #ret,frame = cap.read()
    if ret:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        #p1 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
        
        p3 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)                
        print("[p3 " + str(len(p3)) + "] [p0 " + str(len(p0)) + "]")

        # check if there are any new good feature...
        


        #print(p1)
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        # draw the tracks
        for k,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), color[k].tolist(), 2)
            frame = cv2.circle(frame,(a,b),5,color[k].tolist(),-1)
            #img = cv2.add(frame, mask)
            img = cv2.add(mask, frame)
            cv2.imshow('frame',mask)
            #cv2.imshow('frame', img)
            
            out.write(mask)
            i = ( i + 1 ) % 4
            #print(progressArray[i])    
            #sys.stdout.write('\rprocessing frames... {0} '.format(progressArray[i]))
            #sys.stdout.flush()
            
            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1,1,2)
        pause = True

cap.release()
out.release()
cv2.destroyAllWindows()
print()
