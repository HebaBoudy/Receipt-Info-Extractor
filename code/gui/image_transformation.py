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
from common_functions import fix_image_type

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


def getThreshold(image: np.ndarray):
    # gray_scaled_image = image
    # if (image.ndim >= 3):
    #     gray_scaled_image = rgb2gray(image)
    # gray_scaled_image = (gray_scaled_image * 255).astype(np.uint8)
    # hist = custom_histogram(gray_scaled_image)
    hist, _ = np.histogram(image, bins=256, range=(0, 256))
    
    total_number_of_pixels = np.sum(hist)
    numerator = np.sum(np.array([(hist[i] * i) for i in range(256)]))
    threshold_old = int(np.round(numerator / total_number_of_pixels))

    while True:
        lower_numerator = np.sum(np.array([(hist[i] * i) for i in range(threshold_old)]))
        upper_numerator = np.sum(np.array([(hist[i] * i) for i in range(threshold_old, 256)]))
        lower_pixel_mean = np.round(lower_numerator / np.sum(hist[:threshold_old]))
        upper_pixel_mean = np.round(upper_numerator / np.sum(hist[threshold_old:]))

        threshold = int((lower_pixel_mean + upper_pixel_mean) / 2)
        if threshold == threshold_old:
            break
        threshold_old = threshold
    return threshold


def local_thresholding(image: np.ndarray, splits: int, axis: int = 0):
    paritions = np.split(image, splits, axis=axis)
    thresholds = [getThreshold(partition) for partition in paritions]
    thresholded_partitions = [partition > threshold for partition, threshold in zip(paritions, thresholds)]
    image_thresholded = np.concatenate(thresholded_partitions, axis=axis)
    return image_thresholded

# Source: https://stackoverflow.com/questions/46390779/automatic-white-balancing-with-grayworld-assumption
def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def gamma_correction(img: np.ndarray, gamma):
    img = fix_image_type(img)
    img = exposure.adjust_gamma(img, gamma)
    return img