

import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import histogram
from matplotlib.pyplot import bar
from skimage.color import rgb2gray,rgb2hsv
from skimage.morphology import   binary_erosion, binary_dilation, binary_closing, binary_opening
# Convolution:
from scipy.signal import convolve2d
from scipy import fftpack
import math
import cv2

from skimage.util import random_noise
from skimage.filters import median, threshold_otsu
from skimage.feature import canny

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

# Edges
from skimage.filters import sobel_h, sobel, sobel_v,roberts, prewitt

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


def contrast_enhancement(image: np.ndarray, with_plot=False):
    if image.ndim > 2:
        r_image, g_image, b_image = cv2.split(image)

        r_image_eq = cv2.equalizeHist(r_image)
        g_image_eq = cv2.equalizeHist(g_image)
        b_image_eq = cv2.equalizeHist(b_image)

        image_eq = cv2.merge((r_image_eq, g_image_eq, b_image_eq))
        cmap_val = None
    else:
        image_eq = cv2.equalizeHist(image)
        cmap_val = 'gray'

    if with_plot:
        fig = plt.figure(figsize=(10, 20))

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.axis("off")
        ax1.title.set_text('Original')
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.axis("off")
        ax2.title.set_text("Equalized")

        ax1.imshow(image, cmap=cmap_val)
        ax2.imshow(image_eq, cmap=cmap_val)
        return True
    return image_eq

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
