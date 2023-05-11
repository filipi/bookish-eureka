#!/usr/bin/env python

'''
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.

Usage
-----
lk_track.py [<video_source>]


Keys
----
ESC - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
import video
from common import anorm2, draw_str
#from time import clock

import cmath

from scipy.signal import argrelextrema

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )


#new_tracks = []

def goodCleftsToTrack(frame_gray):
    _, thresh = cv2.threshold(frame_gray,127,255,cv2.THRESH_BINARY_INV)
    contours,hierachy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    tamanhao = 0
    ## contours tem um array com vários outros arrays da árvore de contornos
    ## Aqui soma os tamanhos de todos os arrays contendo contornos
    for i in range(len(contours)):
        tamanhao = tamanhao + len(contours[i])
        # tamanhao tem o número de pontos que tem o contorno detectado
    b = [0] * tamanhao # cria uma lista (array python) b com o tamanho do total de pontos do contorno detectado
    k = 0
    for j in range(len(contours)):
        for i in range(len(contours[j])):
            #print(contours[3][i][0][0], contours[3][i][0][1])        
            b[k] = list(contours[j][i][0]) ## copia os pontos de contours para b
            k = k + 1      
    a=np.array(b)            # copia o conteudo da lista b para um np.array a

    #print("Shape of the array (frame_gray) = ",np.shape(frame_gray));
    #print("Shape of the array (contours) = ",np.shape(contours));
    #print("Shape of the array (b) = ",np.shape(b));
    #print("Shape of the array (a) = ",np.shape(a));
    
    a=a[np.argsort(a[:, 1])] # ordena o a em funcao da coluna 1
    minimos = argrelextrema(a[:,0], np.less, order = 2)
        
    x = a[:,1]
    y = a[:,0]
        
    p = np.array(list(zip(y[minimos],x[minimos])))
    #print("Shape of the array (p) = ",np.shape(p));


    #if np.any(contours) == False:
    #    print(p)
    #    print("Shape of the array (p) = ",np.shape(p));
    #    #return p    
    
    return p
    #p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
        
        

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)
    
def goodCleftsToTrackPolar(frame_gray):
    _, thresh = cv2.threshold(frame_gray,127,255,cv2.THRESH_BINARY_INV)
    contours,hierachy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    height = frame_gray.shape[0] ## acha a altura da imagem para inversoes translacoes
   
    tamanhao = 0
    ## contours tem um array com vários outros arrays da árvore de contornos
    ## Aqui soma os tamanhos de todos os arrays contendo contornos
    for i in range(len(contours)):
        tamanhao = tamanhao + len(contours[i])
        # tamanhao tem o número de pontos que tem o contorno detectado
    b = [0] * tamanhao # cria uma lista (array python) b com o tamanho do total de pontos do contorno detectado
    k = 0
    for j in range(len(contours)):
        for i in range(len(contours[j])):
            #print(contours[3][i][0][0], contours[3][i][0][1])        
            b[k] = list(contours[j][i][0]) ## copia os pontos de contours para b
            k = k + 1
            
    a=np.array(b)            # copia o conteudo da lista b para um np.array a
    
    a[:,1]=np.invert(a[:,1])+height
    print(np.size(a[:,1]))

    c = np.zeros(shape=(len(a),2))
    d = np.zeros(shape=(len(a),2))
    d[:,0]=a[:,0]
    d[:,1]=a[:,1]
    a_polar = cart2pol(d[:,0], d[:,1])
    c[:,1]=a_polar[1]
    c[:,0]=a_polar[0]
    c=c[np.argsort(c[:, 1])]
    xLine = c[:,0]
    yLine = c[:,1]
    minimos = argrelextrema(xLine, np.less, order = 10)
    a_cart = pol2cart(xLine[minimos], yLine[minimos])
    p = np.array(list(zip(a_cart[0], a_cart[1])))
    p[:,1] = p[:,1]-height
    p[:,1] = np.invert(p[:,1].astype(int))
    
    return p
    #p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)

class App:
    def __init__(self, video_src):
        self.track_len = 10
        self.detect_interval = 1
        self.tracks = []
        self.cam = video.create_capture(video_src)
        self.frame_idx = 0        

    def run(self):
        ret, frame = self.cam.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        total = frame_gray
        vis = frame.copy()        
        vis_gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)                    
        trail_mask = np.zeros_like(vis_gray)
        trail_mask = (255 - trail_mask)        
        trail_channels = trail_mask        
        
        output_frame_idx=0

        while True:
            ch = cv2.waitKey(1)
            #ch = 32 ## uncoment to force not to wait
            
            if ch == 27:
                break

            if ch == 32:
                ret, frame = self.cam.read()

                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                vis = frame.copy()
                

                if len(self.tracks) > 0:
                    img0, img1 = self.prev_gray, frame_gray
                    p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                    p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                    p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                    d = abs(p0-p0r).reshape(-1, 2).max(-1)
                    good = d < 1
                    new_tracks = []
                    #print("passei new") ## O problema é nas imagens, que não gera corretamente
                    for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                        if not good_flag:
                            continue
                        tr.append((x, y))
                        if len(tr) > self.track_len:
                            del tr[0]
                        new_tracks.append(tr)
                        cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
                    
                    self.tracks = new_tracks
                    #print("passei assign")
                    
                    cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 0, 0))
                    cv2.polylines(trail_channels, [np.int32(tr) for tr in self.tracks], False, (0, 0, 0))                    
                    #draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))
                    #draw_str(trail_channels, (20, 20), 'track count: %d' % len(self.tracks))
                    
                    #print('track count: %d ' % len(self.tracks))

                    if (len(self.tracks)<10):
                        print('frame: %d ' % self.frame_idx)
                        print('track count: %d ' % len(self.tracks))
                        print('')

                if self.frame_idx % self.detect_interval == 0:
                    mask = np.zeros_like(frame_gray)
                    mask[:] = 255
                    for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                        cv2.circle(mask, (x, y), 5, 0, -1)
                        
                    ###################### Aqui que se escolhe o metodo ###
                    #p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                    #p = goodCleftsToTrack(frame_gray)
                    p = goodCleftsToTrackPolar(frame_gray)
                    #######################################################
                    
                    if p is not None:
                        for x, y in np.float32(p).reshape(-1, 2):
                            self.tracks.append([(x, y)])

                vis_gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
                trail_mask = np.bitwise_and(trail_mask, trail_channels)#<<<<<<<<<<                    
                ##trail_mask = np.bitwise_and(trail_mask, vis_gray)#<<<<<<<<<<                    
                ##print(vis_gray.shape)
                #print(output_frame_idx)
                output_frame_idx = output_frame_idx + 1

            self.frame_idx += 1            
            # Descomentar para usar overlay
            #total = total & frame_gray
            self.prev_gray = frame_gray

            # Uncoment to save images
            fn = './resultados/tracks_%04d.png' % self.frame_idx
            #cv2.imwrite(fn, trail_mask)
            cv2.imwrite(fn, total & trail_mask)
            
            ##cv2.imshow('lk_track', vis)
            cv2.imshow('lk_track', total & trail_mask)            
            ###cv2.imshow('lk_track', trail_mask)
            ###cv2.imshow('lk_track', trail_channels)
            
                


def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    print(__doc__)
    App(video_src).run()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
