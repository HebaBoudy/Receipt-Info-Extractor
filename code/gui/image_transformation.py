import os

import cv2
import numpy as np
import math
from skimage.filters import threshold_minimum,threshold_otsu
from skimage.exposure import equalize_hist
from scipy.ndimage import binary_erosion,binary_closing,binary_opening,binary_dilation
from skimage import exposure
from skimage.measure import find_contours
from skimage.draw import rectangle
from PIL import Image


def Erode(img,windowSize):
    window=np.ones((windowSize,windowSize))
    Final=np.zeros(img.shape)
    for i in range(img.shape[0]-windowSize+1):
        for j in range(img.shape[1]-windowSize+1):
            Fit=np.all(np.logical_and(window,img[i:i+windowSize,j:j+windowSize]))
            Final[i,j]=Fit
    return Final

def Dilate(img,windowSize):
    window=np.ones((windowSize,windowSize))
    Final=np.zeros(img.shape)
    for i in range(img.shape[0]-windowSize+1):
        for j in range(img.shape[1]-windowSize+1):
            Hit=np.any(np.logical_and(window,img[i:i+windowSize,j:j+windowSize]))
            Final[i,j]=Hit
    return Final 
def custom_erosion (image , window_size ):
    height ,  width  = image.shape 
    edge_y = window_size[0] // 2
    edge_x = window_size[1] // 2 
    window = np.ones(window_size) 
    out_image_erosion = np.zeros(image.shape)

    for y in range(edge_y , height - edge_y) :
        for x in range(edge_x , width - edge_x) :
            image_window = image [y-edge_y:y+edge_y+1,x-edge_x:x+edge_x+1] 
            multiplied_img = image_window*window
            out_image_erosion[y,x] = np.all(multiplied_img == 1) 
    return out_image_erosion 


def custom_dilation (image , window_size ):
    height ,  width  = image.shape 
    edge_y = window_size[0] // 2
    edge_x = window_size[1] // 2 
    window = np.ones(window_size) 
    out_image_dilation = np.zeros(image.shape)
    for y in range(edge_y , height - edge_y) :
        for x in range(edge_x , width - edge_x) :
            image_window = image [y-edge_y:y+edge_y+1,x-edge_x:x+edge_x+1] 
            multiplied_img = image_window*window
            out_image_dilation[y,x] = np.any(multiplied_img == 1) 
    return out_image_dilation
def Gamma_correction(img,gamma):
    img = img / img.max()
    img = img* 255
    img = img.astype(np.uint8)
    img = exposure.adjust_gamma(img, gamma)
    return img

def houghTransform(img):
    orgimg=img.copy()
    cdst = orgimg.copy()
    cdstP = np.copy(cdst)
    
    lines = cv2.HoughLines(orgimg, 1, np.pi / 180, 150, None, 0, 0)
    
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(cdst, pt1, pt2, (255,255,255), 2, cv2.LINE_AA)
    
    
    linesP = cv2.HoughLinesP(orgimg, 1, np.pi / 180, 50, None, 50, 60)
    
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (255,255,255), 2, cv2.LINE_AA)
    
    return cdstP

# Window size can be rectangle (not equal sides)
def erosion(image: np.ndarray, size_y: int = 3, size_x: int = 3):
    window = np.ones((size_y, size_x))
    rows, cols = image.shape
    edge_x = size_x // 2
    edge_y = size_y // 2
    size_y_odd, size_x_odd = size_y % 2 != 0, size_x % 2 != 0
    threshold = threshold_otsu(image)
    binary_img = image > threshold 

    output_img = np.zeros(binary_img.shape)
    for row in range(edge_y, rows - edge_y):
        for col in range(edge_x, cols - edge_x):
            output_img[row, col] = np.all(binary_img[row - edge_y: row + edge_y + size_y_odd, col - edge_x: col + edge_x + size_x_odd] * window)

    return output_img

# Window size can be rectangle (not equal sides)
def dialation(image: np.ndarray, size_y: int = 3, size_x: int = 3):
    window = np.ones((size_y, size_x))
    rows, cols = image.shape
    edge_x = size_x // 2
    edge_y = size_y // 2
    size_y_odd, size_x_odd = size_y % 2 != 0, size_x % 2 != 0
    threshold = threshold_otsu(image)
    binary_img = image > threshold 

    output_img = np.zeros(binary_img.shape)
    for row in range(edge_y, rows - edge_y):
        for col in range(edge_x, cols - edge_x):
            output_img[row, col] = np.any(binary_img[row - edge_y: row + edge_y + size_y_odd, col - edge_x: col + edge_x + size_x_odd] * window)

    return output_img