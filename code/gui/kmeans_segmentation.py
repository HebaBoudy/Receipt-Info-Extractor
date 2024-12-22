import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.measure import find_contours
from scipy.ndimage import binary_erosion, binary_closing, binary_opening, binary_dilation

def find_reciept_kmeans(image: np.ndarray):
    """
    Find the receipt in the image using K-Means segmentation.
    Args:
        image: Uploaded receipt image.
    Returns:
        Extracted receipt image.
    """
    # Perform K-Means segmentation
    labels, centers = kmeans(image, 2)

    white_label = np.argmax(np.sum(centers, axis=1))  # Identify the white region
    mask = (labels == white_label)
    segmented_mask = mask.reshape(image.shape[:2])

    # Apply mask to the original image
    segmented_image = cv2.bitwise_and(image, image, mask=segmented_mask.astype(np.uint8))

    segmentedImageRgb = rgb2gray(segmented_image)
    binarySeg = basel_thresholding(segmentedImageRgb)

    opened = binary_opening(binarySeg,np.ones((2,2)), iterations=15) #problem 2
    # Find the contours of the receipt
    contours = find_contours(opened, 0.8)
    contours = [np.array(contour, dtype=np.int32) for contour in contours]
    maxC = max(contours, key=cv2.contourArea)
    c = np.array([[p[1], p[0]] for p in maxC], dtype=np.int32)
    reciept_contour = approximate_to_rectangle(c)

    input_points = find_polygon_corners(reciept_contour)
    input_points = fix_outlier_point_with_distance(input_points)

    receipt = transform_perspective(image, input_points)
    receiptGrey = rgb2gray(receipt)
    return receipt
    return receiptGrey
    

def kmeans(image_rgb,k,randomSeed = 42):
    cv2.setRNGSeed(randomSeed)

    # Reshape the image into a 2D array of pixels and 3 color values (RGB)
    pixel_values = image_rgb.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Define criteria, number of clusters (K), and apply K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    #k = 2  # Number of clusters
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert centers to uint8 (RGB values)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image_rgb.shape)

    # Create a mask for the cluster corresponding to the statue
    l = labels.copy()
    labels = labels.flatten()
    mask = (labels == 0)  # Adjust based on which cluster corresponds to the statue
    segmented_mask = mask.reshape(image_rgb.shape[:2])

    # Apply mask to the original image
    statue_segmented = cv2.bitwise_and(image_rgb, image_rgb, mask=segmented_mask.astype(np.uint8))

    return labels, centers


def basel_thresholding(image: np.ndarray):
    threshold = 0.6 * image.max()
    mask = image > threshold
    binarySeg = np.zeros(image.shape)
    binarySeg[mask] = 255
    binarySeg = binarySeg.astype(np.uint8)
    return binarySeg


def approximate_to_rectangle(c, max_iterations=100000, initial_epsilon_factor=0.09):
    """Iteratively approximates a contour to a rectangle with 4 corners."""
    epsilon_factor = initial_epsilon_factor
    for _ in range(max_iterations):
        peri = cv2.arcLength(c, True)
        approx_polygon = cv2.approxPolyDP(c, epsilon_factor * peri, True)
        
        # Check if the polygon has 4 corners
        if len(approx_polygon) == 4:
            return approx_polygon  # Successfully approximated to a rectangle
        
        # Reduce epsilon for finer approximation
        epsilon_factor *= 0.9  # Reduce by 10% each iteration
    
    # If no rectangle is found after max_iterations, return the last approximation
    return None  # Or return the last `approx_polygon` if you want

#top-left, top-right, bottom-right, bottom-left.
def find_polygon_corners(reciept_contour):
    """Identifies the corners of the detected polygon."""
    points = reciept_contour.reshape(4, 2)
    input_points = np.zeros((4, 2), dtype="float32")
    points_sum = points.sum(axis=1)
    input_points[0] = points[np.argmin(points_sum)]
    input_points[3] = points[np.argmax(points_sum)]
    points_diff = np.diff(points, axis=1)
    input_points[1] = points[np.argmin(points_diff)]
    input_points[2] = points[np.argmax(points_diff)]
    #show points on the image

    return input_points


def fix_outlier_point_with_distance(input_points):
    """
    Fixes an outlier point in the array of 4 points (top-left, top-right, bottom-right, bottom-left).
    Adjusts the outlier by reducing its distance from the centroid to align with the mean distance.
    """
    if len(input_points) != 4:
        raise ValueError("Input must contain exactly 4 points.")

    # Step 1: Calculate the centroid (center of all points)
    centroid = np.mean(input_points, axis=0)

    # Step 2: Calculate the distances of each point from the centroid
    distances = np.linalg.norm(input_points - centroid, axis=1)

    # Step 3: Identify the outlier point
    outlier_index = np.argmax(distances)

    # Step 4: Calculate the mean distance of the other three points
    valid_distances = np.delete(distances, outlier_index)
    mean_distance = np.mean(valid_distances)

    # Step 5: Calculate the difference and adjust the outlier
    distance_diff = distances[outlier_index] - mean_distance

    # Adjust the outlier coordinates proportionally
    outlier_point = input_points[outlier_index]
    direction_vector = outlier_point - centroid
    normalized_vector = direction_vector / np.linalg.norm(direction_vector)
    correction = normalized_vector * distance_diff
    corrected_point = outlier_point - correction

    # Step 6: Replace the outlier point with the corrected point
    input_points[outlier_index] = corrected_point

    return input_points

def transform_perspective(image, input_points):
    """Performs a perspective transformation on the image."""
    max_width = 650
    max_height = 1024
    converted_points = np.float32([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]])
    matrix = cv2.getPerspectiveTransform(input_points, converted_points)
    img_output = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return img_output
