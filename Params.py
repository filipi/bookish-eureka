import numpy as np
import cv2


class Params:
    erodeKernel = None
    dilateKernel = None 
    erodeIterations = None
    dilateIterations = None
    thresholdMin = None
    thresholdMax = None
    epsilon = None
    criteriaMaxIterations = None
    subPixCriteria = None

class Linear (Params):
    erodeKernel = np.ones((3, 3), np.uint8)
    dilateKernel = np.ones((3, 3), np.uint8)
    erodeIterations = 1
    dilateIterations = 1
    thresholdMin = 190
    thresholdMax = 255
    epsilon = 0.005
    subPixCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

class Polar (Params):
    erodeKernel = np.ones((3, 3), np.uint8)
    dilateKernel = np.ones((3, 3), np.uint8)
    erodeIterations = 1
    dilateIterations = 1
    thresholdMin = 190
    thresholdMax = 255
    epsilon = 0.0008
    subPixCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)

class Testes (Params):
    erodeKernel = np.ones((3, 3), np.uint8)
    dilateKernel = np.ones((3, 3), np.uint8)
    erodeIterations = 1
    dilateIterations = 1
    thresholdMin = 190
    thresholdMax = 255
    epsilon = 0.0035
    subPixCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)