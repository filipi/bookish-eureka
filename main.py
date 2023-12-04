import cv2
import numpy as np
from scipy import signal, spatial
from Params import Linear, Params

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def getContours(frameGray, params:Params):
    smoothed_contours = cv2.erode(frameGray, params.erodeKernel, iterations=params.erodeIterations)
    smoothed_contours = cv2.dilate(smoothed_contours, params.dilateKernel, iterations=params.dilateIterations)
    _, frameThreshold = cv2.threshold(smoothed_contours, params.thresholdMin, params.thresholdMax, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(frameThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

def goodCleftsToTrack(frameGray, contours, params:Params):
    cleftsInFrame = []
    for contour in contours:
        epsilon = params.epsilon * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        x_values = [pt[0][0] for pt in approx]
        local_minima_indices = signal.argrelextrema(np.array(x_values), np.less)[0]
        local_minima_points = [approx[i] for i in local_minima_indices]
        if local_minima_points.__len__()> 0:
            cleftsInFrame = cleftsInFrame + local_minima_points 
    cleftsInFrame = np.float32(list(cleftsInFrame)).reshape(-1,2)
    cleftsInFrame = cv2.cornerSubPix(frameGray, cleftsInFrame, (5,5), (-1,-1), params.subPixCriteria)
    return cleftsInFrame

def main():
    fileName = 'LR_Re8950'
    cap = cv2.VideoCapture(f'obtaningData\\{fileName}.mp4')

    lastFrameGray = None
    frameTracks = None
    lastOptical = None
    frameIdx = 0
    while True:
        success, frame = cap.read()
        if not success:
            break 

        frameIdx = frameIdx + 1

        # Inicialização do Frame das Linhas
        if frameTracks is None:
            frameTracks = frame.copy()
            frameReturn = np.zeros_like(frame)
            
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frameContour = frame.copy()
        frameOptical = frame.copy()
        frameMinima = frame.copy()

        # Detecção da linha de contorno
        contours = getContours(frameGray, Linear)
        cv2.drawContours(frameContour, contours, -1, (255,0,255), 1)
        # Detectar minimos locais
        cleftsInFrame = goodCleftsToTrack(frameGray, contours, Linear)
        for point in cleftsInFrame:
            cv2.circle(frameMinima, tuple((int(point[0]) , int(point[1]) )), 3, (0, 255, 0), -1)

        # Reduz cortes de linha
        if lastOptical is not None:
            cleftTree = spatial.cKDTree(cleftsInFrame)
            opticalTree = spatial.cKDTree(lastOptical)
            distance, indice = opticalTree.query(cleftTree.data)
            for i, j in enumerate(indice):
                if distance[i] > 1:
                    continue
                cv2.line(frameTracks, (int(cleftsInFrame[i][0]), int(cleftsInFrame[i][1])), (int(lastOptical[j][0]), int(lastOptical[j][1])), (255, 0, 0), 1)

        # Calculo de Fluxo Otico
        if len(cleftsInFrame) > 0:
            optical, st, err = cv2.calcOpticalFlowPyrLK(lastFrameGray, frameGray, cleftsInFrame, None, **lk_params)
            #print(f'Optical: {optical}')
            optical2, st2, err2 = cv2.calcOpticalFlowPyrLK(frameGray, lastFrameGray, optical, None, **lk_params)
            #print(f'Optical 2: {optical2}')
            d = abs(cleftsInFrame-optical2).reshape(-1, 2).max(-1)
            goodPoints = d < 2
            # Pontos Futuros
            for tr, (x, y), goodFlag in zip(cleftsInFrame, optical.reshape(-1, 2), goodPoints):
                if not goodFlag:
                    continue
                cv2.circle(frameOptical, (int(x), int(y)), 2, (255, 0,  0), -1)
                # Linhas do ponto anterior até o ótico
                cv2.line(frameTracks, (int(tr[0]), int(tr[1])), (int(x), int(y)), (0, 0, 255), 1)
            lastOptical = optical.reshape(-1, 2)

        lastFrameGray = frameGray

        if frameIdx % 2 == 0:
            frameReturn = cv2.add(frameReturn, cv2.bitwise_not(frameMinima))
            frameReturn = cv2.add(frameReturn, cv2.bitwise_not(frame))
        
        cv2.imshow('Result Minima', frameMinima)
        #cv2.imshow('Result Contour', frameContour)
        cv2.imshow('Result Optical', frameOptical)
        cv2.imshow('Result Lines', frameTracks)
        #cv2.imshow('Combined Lines', frameCombined)
        #cv2.imshow('Result Treshold', frameThreshold)
        cv2.imshow('Result Return', cv2.bitwise_not(frameReturn))

        #if frameIdx % 4 == 0:
        #    cv2.imwrite(f'results\\{fileName}\\frame\\{frameIdx}.png', frame)
        #    cv2.imwrite(f'results\\{fileName}\\minimas\\{frameIdx}.png', frameMinima)
        #    cv2.imwrite(f'results\\{fileName}\\optical\\{frameIdx}.png', frameOptical)

        ch = cv2.waitKey(0)
        if ch == ord('q'):
            break
        elif ch == ord('w'):
            pass 
    ch = cv2.waitKey(0)
    cv2.imwrite(f'results\\{fileName}\\tracks.png ', frameTracks)
    cv2.imwrite(f'results\\{fileName}\\following.png', cv2.bitwise_not(frameReturn))
    cv2.imwrite(f'results\\{fileName}\\following_tracks.png', cv2.bitwise_not(cv2.add(cv2.bitwise_not(frameTracks),frameReturn)))

if __name__ == '__main__':
    main()