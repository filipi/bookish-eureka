import numpy as np
import cv2


class Params:
    erodeKernel = 0
    dilateKernel = 0 
    erodeIterations = 0
    dilateIterations = 0
    thresholdMin = 0
    thresholdMax = 0
    epsilon = 0.0
    criteriaMaxIterations = 0
    subPixCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 0, 0.0)

class Linear (Params):
    erodeKernel = np.ones((3, 3), np.uint8)
    dilateKernel = np.ones((3, 3), np.uint8)
    erodeIterations = 1
    dilateIterations = 1
    thresholdMin = 190
    thresholdMax = 255
    epsilon = 0.0035
    subPixCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

class Polar (Params):
    erodeKernel = np.ones((3, 3), np.uint8)
    dilateKernel = np.ones((3, 3), np.uint8)
    erodeIterations = 1
    dilateIterations = 1
    thresholdMin = 190
    thresholdMax = 255
    epsilon = 0.0021
    subPixCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)