import matplotlib as m
#m.use('Agg') # to plot without X11
m.use('Qt5Agg') # to plot without X11
#http://stackoverflow.com/questions/4931376/generating-matplotlib-graphs-without-a-running-x-server
#https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_shi_tomasi/py_shi_tomasi.html

import numpy as np
import cv2
from matplotlib import pyplot as plt

feature_params = dict( maxCorners = 100, #100,
                       qualityLevel = 0.3, #0.3,
                       minDistance = 7, #7,
                       blockSize = 7 ) #7 )

#endImg  = cv2.imread('obtaningData/Hartel/cropped/01.png')
endImg  = cv2.imread('obtaningData/DNS_IO_5k/cropped/frame1.mini.png')

#for imgIndex in range(1, 71):
for imgIndex in range(0, 136):
    #img  = cv2.imread('obtaningData/Hartel/cropped/' + str(imgIndex).zfill(2) + '.png')
    img  = cv2.imread('obtaningData/DNS_IO_5k/cropped/frame' + str(imgIndex) + '.mini.png')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
    corners = cv2.goodFeaturesToTrack(gray, mask = None, **feature_params)

    corners = np.int0(corners)

    endImg = endImg & img # bitwise and to put all images together, generating and image similar to Hartel's work

    for i in corners:
        x,y = i.ravel()
        cv2.circle(endImg,(x,y),3,255,-1)

plt.imshow(endImg),plt.show()
plt.savefig('test1.png', format='png', dpi=300)  # 'pdf')

plt.close('all')
