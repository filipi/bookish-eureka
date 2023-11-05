import cv2
import numpy as np
import scipy.signal

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


cap = cv2.VideoCapture('obtaningData\LR_Re3450.mp4')

lastFrameGray = None
frameTracks = None
tracks = []

def process_contour(contour):
    # Extract the x-values from the contour
    x_values = [pt[0][0] for pt in contour]
    #print(f'Contornos: {contour}')

    local_minima_indices = scipy.signal.argrelextrema(np.array(x_values), np.less)[0]

    local_minima_points = [contour[i] for i in local_minima_indices]
    return local_minima_points

while True:
    success, frame = cap.read()
    if not success:
        break 

    # Inicialização do Frame das Linhas
    if frameTracks is None:
        frameTracks = frame.copy()
        frameTeste = frame.copy()

    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameContour = frame.copy()
    frameOptical = frame.copy()
    #frameReturn = frame.copy()

    # Detecção da linha de contorno
    _, frameThreshold = cv2.threshold(frameGray, 190, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(frameThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        cv2.drawContours(frameContour, cnt, -1, (255,0,255), 1)
    # Detectar minimos locais
    for contour in contours:
        epsilon = 0.0022* cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        local_minima_points = process_contour(approx)
        #print(f'Local Minimas: {local_minima_points}')
    for point in local_minima_points:
        cv2.circle(frame, tuple(point[0]), 2, (0, 255, 255), -1)  

    # Formata todos os pontos de minima do Frame atual 
    cleftsInFrame = np.float32(list(local_minima_points)).reshape(-1,2)
    #print(f'Clefts In Frame: {cleftsInFrame}')
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    cleftsInFrame = cv2.cornerSubPix(frameGray, cleftsInFrame, (5,5), (-1,-1), criteria)
    #print(f'Clefts In Frame With Criteria: {cleftsInFrame}')
    

    # Calculo de Fluxo Otico
    newTracks = []
    if len(local_minima_points) > 0:
        optical, st, err = cv2.calcOpticalFlowPyrLK(lastFrameGray, frameGray, cleftsInFrame, None, **lk_params)
        #print(f'Optical: {optical}')
        optical2, st2, err2 = cv2.calcOpticalFlowPyrLK(frameGray, lastFrameGray, optical, None, **lk_params)
        #print(f'Optical 2: {optical2}')
        d = abs(cleftsInFrame-optical2).reshape(-1, 2).max(-1)
        goodPoints = d < 1 
        # Pontos Futuros
        for tr, (x, y), goodFlag in zip(cleftsInFrame, optical.reshape(-1, 2), goodPoints):
            if not goodFlag:
                continue
            cv2.circle(frameOptical, (int(x), int(y)), 2, (0, 255,  0), -1)
            # Linhas do ponto anterior até o ótico
            cv2.line(frameTracks, (int(tr[0]), int(tr[1])), (int(x), int(y)), (0, 0, 0), 1)
            # Adiciona linha no array de Tracks
            newTrack = [(int(tr[0]), int(tr[1])), (int(x), int(y))]
            newTracks.append(newTrack)
        #Salva linha em track Global
        tracks.append(newTracks)
        #print(f'Tracks: {tracks}')

    cv2.imshow('Result Minima', frame)
    #cv2.imshow('Result Contour', frameContour)
    cv2.imshow('Result Optical', frameOptical)
    cv2.imshow('Result Lines', frameTracks)
    #cv2.imshow('Combined Lines', frameCombined)
    #cv2.imshow('Result Return', frameTeste)
    #cv2.imshow('Result Treshold', frameThreshold)

    lastFrameGray = frameGray

    ch = cv2.waitKey(0)
    if ch == ord('q'):
        break
    elif ch == ord('w'):
        pass