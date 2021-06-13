#!/usr/bin/env python

#https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html
import numpy as np
import cv2
import sys

feature_params = dict( maxCorners = 100, #100, # params for ShiTomasi corner detection
                       qualityLevel = 0.2, #0.3,,#0.2,
                       minDistance = 7, #12, #7,
                       blockSize = 7)# 7 ) #7 ) #12 )
lk_params = dict( winSize  = (50, 50),#(200,200) #(15,15), # Parameters for lucas kanade optical flow
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.03))
color = np.random.randint(0,255,(100,3)) # Create some random colors
#color = (0, 0, 255) 
#color_of = (0, 255, 0)

pause = False
frameNumber = 0

i = 0
progressArray = ['-', '\\', '|', '/' ]

structures = []
corners = np.ndarray([])

#######################################################################################################

#capFileName = '../obtaningData/basin_DNS.avi'
#apFileName = '../obtaningData/simpson_1972_small.mpg'
#capFileName = '../obtaningData/simpson_1972_fast.mpg'
#capFileName = '../obtaningData/Simpson/frontSimpson.mpg'
capFileName = '../obtaningData/Neufeld/neufeld.mpg'
#capFileName = '../obtaningData/lockExchangeSimpson_filipi_Re2445/test.mp4'
#capFileName = '../frenteSimpson.mpg'
#capFileName = '../obtaningData/Mariana/mariana.mp4'

#######################################################################################################


cap = cv2.VideoCapture(capFileName)
# Take first frame and find corners in it
for i in range(0, 2):
    ret, old_frame = cap.read()


ret, old_frame = cap.read()
mask = np.zeros_like(old_frame)
mask = (255-mask)
frame = old_frame

cv2.imshow('frame', frame)

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY) 
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)               

while(1):
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    if k == 32:
        
        # (frameNumber)
        frameNumber = frameNumber + 1   
        
        #descomentar aqui para redescobrir os cantos
        #old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY) 
        #p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)               
        
        old_gray_test = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY) 
        p3 = cv2.goodFeaturesToTrack(old_gray_test, mask = None, **feature_params)               
        
        
        frame_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        #p1 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)        
        #print(p1)
        #print(p0)        
        #break
        
        #print(p3)
        
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]        
        corner_new = p3.reshape(-1,2)  # esses novos cantos vao vir em numero diferente e nao da pra usar o 
                                       # indice condicional st, mas precisa dar o reshpe para ficar um array
                                       # de uma posicao com um outro array dentro comos pontos e nao varios arrays
                                       # com os pontos
        print(corner_new.size)
        #print(good_old)        
        #print(corner_new)
        #break
        
        # draw the tracks
        for k,(corner) in enumerate(corner_new):
            e,f = corner.ravel()
            frame2 = cv2.circle(old_frame,(e,f),5,color[k].tolist(),-1)
            cv2.imshow('frame2', frame2)
            
        for k,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()            
            
            #print(a)
            #quit()
            mask = cv2.line(mask, (a,b),(c,d), color[k].tolist(), 2)
            #mask = cv2.line(old_frame, (a,b),(c,d), color[k].tolist(), 5)
            frame = cv2.circle(old_frame,(a,b),5,color[k].tolist(),-1)
            
            
            
            #mask = cv2.line(mask, (a,b),(c,d), color_of, 5)
            #frame = cv2.circle(old_frame,(a,b),10,color,-1)           
            
            #img = cv2.add(frame, mask)
            #mask = mask + frame
            mask = np.bitwise_and(mask, frame)#<<<<<<<<<<
            cv2.imshow('mask', mask)
            cv2.imshow('frame', frame)
            
            
            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1,1,2)
            #break           
            
            i = ( i + 1 ) % 4
            #print(progressArray[i])    
            sys.stdout.write('\rprocessing frames...[{0}] - {1} {2} '.format(frameNumber, k, progressArray[i]))
            sys.stdout.flush()
            
            
        ret, old_frame = cap.read()                            
        structures.append(k)
        if old_frame is None:
            #print(frameNumber)
            break

cv2.destroyAllWindows()
cap.release()
