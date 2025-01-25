import os
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import bar
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import skimage.io as io
from skimage import exposure
from skimage.exposure import histogram, equalize_hist
from skimage.color import rgb2gray, rgb2hsv
from skimage.morphology import binary_erosion, binary_dilation, binary_closing, binary_opening
from skimage.measure import find_contours
from skimage.draw import rectangle
from skimage.util import random_noise
from skimage.filters import median, threshold_otsu, threshold_minimum
from skimage.feature import canny
from skimage.filters import sobel_h, sobel, sobel_v, roberts, prewitt

from PIL import Image

# Convolution:
from scipy.signal import convolve2d
from scipy import fftpack



# Show the figures / plots inside the notebook
def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        plt.axis('off')
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show() 
    

def show_3d_image(img, title):
    fig = plt.figure()
    fig.set_size_inches((12,8))
    ax = fig.gca(projection='3d')

    # Make data.
    X = np.arange(0, img.shape[0], 1)
    Y = np.arange(0, img.shape[1], 1)
    X, Y = np.meshgrid(X, Y)
    Z = img[X,Y]

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(0, 8)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_title(title)
    plt.show()
    
def show_3d_image_filtering_in_freq(img, f):
    img_in_freq = fftpack.fft2(img)
    filter_in_freq = fftpack.fft2(f, img.shape)
    filtered_img_in_freq = np.multiply(img_in_freq, filter_in_freq)
    
    img_in_freq = fftpack.fftshift(np.log(np.abs(img_in_freq)+1))
    filtered_img_in_freq = fftpack.fftshift(np.log(np.abs(filtered_img_in_freq)+1))
    
    show_3d_image(img_in_freq, 'Original Image')
    show_3d_image(filtered_img_in_freq, 'Filtered Image')


def showHist(img):
    # An "interface" to matplotlib.axes.Axes.hist() method
    plt.figure()
    imgHist = histogram(img, nbins=256)
    
    bar(imgHist[1].astype(np.uint8), imgHist[0], width=0.8, align='center')





def crop_image(image: np.ndarray, x: float, y: float, width: float, height: float):
    return image[y : y+height, x : x+width].copy()

def convert_to_binary(image: np.ndarray):
    threshold = threshold_otsu(image)
    mask = image > threshold
    binary_image = np.zeros(image.shape)
    binary_image[mask] = 255
    binary_image = binary_image.astype(np.uint8)
    return (image > threshold).astype(np.uint8)

# Check if x is between y and z
is_between = lambda x, y, z: y <= x <= z

def fix_image_type(image: np.ndarray):
    if image.dtype == np.uint8:
        return image
    return (image * 255).astype(np.uint8)

def add_padding(x: float, y: float, width: float, height: float, padding_x: float, padding_y: float):
    return x - padding_x // 2, y - padding_y // 2, width + padding_x, height + padding_y

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
