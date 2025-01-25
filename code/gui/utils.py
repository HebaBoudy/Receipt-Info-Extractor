import re
import cv2
import pytesseract
import numpy as np
from PIL import Image
from skimage.feature import canny
from common_functions import *
from kmeans_segmentation import *
from skimage.util import img_as_ubyte

def process_receipt_image(image, pipeline_option):
    """
    Process the receipt image and extract data using OCR.
    Args:
        image: Uploaded receipt image.
    Returns:
        Dictionary with extracted data.
    """
    try:
        image_path = 'imgs/'+ image.name
        print(image_path, os.path.dirname(os.path.abspath(__file__)))
        if not os.path.exists(image_path):
            image = image.name.split('.')[0] + ".jpeg"
        image = load_image(image)

        digits = None
        price = None

        if pipeline_option == "Pipeline 1: K-means + Edge Detection":
            # Apply K-means clustering and edge detection
            reciept = find_reciept(image)
            digits = find_digits(reciept)
            splited_digits = split_digits(digits)
            splited_digits = [rgb2gray(digit) if digit.ndim == 3 else digit for digit in splited_digits]
            predicted_digits = post_template_matching(splited_digits)
            digits = predicted_digits
            try:
                price = get_price(rgb2gray(reciept))
            except:
                price = None

        elif pipeline_option == "Pipeline 2: K-means Only":
            # Apply K-means clustering only
            reciept ,reciept_gray= find_reciept_kmeans(image)
            digits = find_digits_kmeans_approach(reciept_gray) 
            try:
                price = get_price(reciept_gray)
            except:
                price = None

        elif pipeline_option == "Pytesseract":
            # Use the raw image for Pytesseract
            reciept ,reciept_gray= find_reciept_kmeans(image)

            # Preprocess the image (grayscale, thresholding)
            _, binary = cv2.threshold(reciept_gray, 128, 255, cv2.THRESH_BINARY)
            
            # Perform OCR
            raw_text = pytesseract.image_to_string(img_as_ubyte(binary))
            
            print("raw_text")
            print(raw_text)
            
            digits = re.search(r"(\d{4}\s){3}\d{4}", raw_text)
            digits = digits.group().strip() if digits else None

            price = re.search(r".+EGP", raw_text)
            if price:
                price = price.group().strip()
                price = re.sub(r"\s*[;,]\s*", ".", price)


        # Parse the OCR output (example parsing logic)
        data = {}
        data["Digits"] = digits
        data["Price"] = price if price else None
        
        print(data)
        return data
    except Exception as e:
        print(f"Error: {e}")
        return None


#* Load Image
def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from the specified path and fix its orientation.
    orientation values:
        # 1 = Horizontal (normal)
        # 2 = Mirror horizontal
        # 3 = Rotate 180
        # 4 = Mirror vertical
        # 5 = Mirror horizontal and rotate 270 CW
        # 6 = Rotate 90 CW
        # 7 = Mirror horizontal and rotate 90 CW
        # 8 = Rotate 270 CW
    Parameters:
    image_path (str): Path to the image file.
    
    Returns:
    numpy.ndarray: Image array.
    """
    image = Image.open(image_path)
    exif = image._getexif()  # Get EXIF data

    if exif:
        orientation = exif.get(274)  # Orientation tag is 274
        if orientation:
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
            elif orientation == 2:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 4:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            elif orientation == 5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT).rotate(270, expand=True)
            elif orientation == 7:
                image = image.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True)
    return np.array(image)


#* Find Reciept
def preprocess_image(image):
    """Loads the image, converts it to grayscale, and applies gamma correction."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = gamma_correction(gray, 1.5)
    # gray_image = cv2.equalizeHist(gray)
    return gray, gray_image


def threshold_and_edge_detection(gray_image):
    """Applies thresholding and edge detection to preprocess the image for contour detection."""
    # image_thresholded = local_thresholding(gray_image, 64)
    # show_images([gray_image, image_thresholded], ['Original Image', 'Thresholded Image'])
    
    threshold = threshold_minimum(gray_image)
    mask = gray_image > threshold
    binary_image = np.zeros(gray_image.shape)
    binary_image[mask] = 255
    binary_image = binary_image.astype(np.uint8)
    # sigma=50
    edges = canny(gray_image, sigma=40, low_threshold=0.5, high_threshold=0.6)
    return custom_dilation(edges, 5)


def filter_reciept_contours(image: np.ndarray, contours):
    """Filters contours to find a rectangular contour with 4 corners and a large area."""
    # rectangular_contours = []
    
    if image.ndim == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
    image_area = height * width

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    max_area = 0
    reciept_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        #if area > max_area:
        sub_epsilon = 0.015
        peri = cv2.arcLength(contour, True)
        approx_polygon = cv2.approxPolyDP(contour, sub_epsilon * peri, True)
        print(len(approx_polygon))
        
        trial = 0
        max_trials = 3
        while len(approx_polygon) > 4 and trial < max_trials:
            peri = cv2.arcLength(contour, True)
            approx_polygon = cv2.approxPolyDP(contour, sub_epsilon * peri, True)
            print(len(approx_polygon))
            sub_epsilon += 0.005
            trial += 1
            
        if len(approx_polygon) == 4 or area < 0.5 * image_area:
            reciept_contour = approx_polygon
            max_area = area
            print("Area: ", max_area)
            break

    return reciept_contour, max_area

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
    return input_points

def transform_perspective(image, input_points):
    """Performs a perspective transformation on the image."""
    max_width = 650
    max_height = 1024
    converted_points = np.float32([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]])
    matrix = cv2.getPerspectiveTransform(input_points, converted_points)
    img_output = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return img_output

def find_reciept(image):
    """Main function to detect a reciept in an image and return the transformed perspective."""
    
    segmented_image, opening = segment_receipt_by_colors(image)
    gray, gray_image = preprocess_image(segmented_image) # Convert to grayscale and apply gamma correction
    opening_edges = sobel(opening) # Canny was not working well
    show_images([segmented_image, opening, opening_edges], ["Segmented Image", "Opening", "Edges"])
    
    opening_contours, _ = cv2.findContours(fix_image_type(opening_edges), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    opening_reciept_contour, opening_area = filter_reciept_contours(opening_edges, opening_contours)

    # If opening is not working (failed to get a contour with 4 corners), try segmented_image
    if not opening_reciept_contour is None and len(opening_reciept_contour) > 4:
        seg_edges = sobel(gray_image)
        seg_contours, _ = cv2.findContours(fix_image_type(seg_edges), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        seg_reciept_contour, seg_area = filter_reciept_contours(seg_edges, seg_contours)
        print(f"seg_area: {seg_area}, opening_area: {opening_area}")
        reciept_contour = seg_reciept_contour if seg_area > opening_area else opening_reciept_contour
    else:
        reciept_contour = opening_reciept_contour

    if reciept_contour is None or len(reciept_contour) == 0 or len(reciept_contour) > 4: # Both failed (segmented and opening)
        return image

    input_points = find_polygon_corners(reciept_contour)

    img_output = transform_perspective(image, input_points)
    return img_output



#* Find Digits
def visualize_contours(image, contours):
    """Visualizes the contours on the image."""
    image = image.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

def find_digits(image: np.ndarray) -> np.ndarray:
    """Main functionality is to detect the 16 digits from the receipt."""
    height, width, _ = image.shape
    image = image.copy()[int(0.3 * height):int(0.8 * height)]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)# * 255
    smoothed_image = cv2.GaussianBlur(gray_image, (5, 5), 0)    

    binary_image = convert_to_binary(smoothed_image)
    binary_image = np.bitwise_not(binary_image)

    dilation_image = dialation(binary_image, 3, 25)
    erosion_image = erosion(dilation_image, 3, 25)
    show_images(
        [image, gray_image, smoothed_image, smoothed_image, binary_image, dilation_image, erosion_image], 
        ["Original", "Gray", "Smoothed", "Local Thresholding", "Binary", "Dilation", "Erosion"]
    )

    dilation_image = fix_image_type(dilation_image)

    # Get the contours of the image
    contours, _ = cv2.findContours(dilation_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, )

    # Filter the contours to get the 16 digits
    digits_contours = None
    largest_area = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        area = cv2.contourArea(contour)

        if is_between(w, 0.6 * width, 0.8 * width) and area > largest_area \
              and not is_between(aspect_ratio, 0.5, 1.5):
            digits_contours = contour
            largest_area = area

    if digits_contours is None:
        return image

    x, y, w, h = cv2.boundingRect(digits_contours)
    x, y, w, h = add_padding(x, y, w, h, padding_x=0, padding_y=4)

    return crop_image(image, x, y, w, h)
