import re
import cv2
import pytesseract
import numpy as np
from PIL import Image
from image_transformation import *
from skimage.feature import canny
from common_functions import *
from kmeans_segmentation import *
from PIL import Image
PITESSERACT = True

def process_receipt_image(image):
    """
    Process the receipt image and extract data using OCR.
    Args:
        image: Uploaded receipt image.
    Returns:
        Dictionary with extracted data.
    """
    try:
        image_path = 'imgs/'+ image.name
        if not os.path.exists(image_path):
            image = image.name.split('.')[0] + ".jpeg"
        image = load_image(image)

        # reciept = find_reciept(image)

        # Using K-Means to segment the image
        reciept ,reciept_gray= find_reciept_kmeans(image)
        digits = None
        # date = None
        price = None
        if(PITESSERACT):

            #* Perform OCR
            # Preprocess the image (grayscale, thresholding)
            _, binary = cv2.threshold(reciept_gray, 128, 255, cv2.THRESH_BINARY)
            binary_pil = Image.fromarray(binary.astype(np.uint8))
            raw_text = pytesseract.image_to_string(binary_pil) 
            print("raw_text")
            print(raw_text)
            digits = re.search(r"(\d{4}\s){3}\d{4}", raw_text)
            digits = digits.group().strip() if digits else None

            # date = re.search(r"\d{2}/\d{2}/\d{4}", raw_text)
            # date = date.group().strip() if date else None
            price = re.search(r".+EGP", raw_text)
            if price:
                price = price.group().strip()
                price = re.sub(r"\s*[;,]\s*", ".", price)
           
        else :
            # digits = find_digits(reciept) 
            # print("after find_digits")
            # splitted_digits = split_digits(digits) 
            # print("after split_digits")
            # digits = post_template_matching(splitted_digits)
            # print(digits)
            digits = find_digits_basel(reciept_gray) 
            price = get_price(reciept_gray)

        # Parse the OCR output (example parsing logic)
        data = {}
        data["Digits"] = digits
        # data["Date"] =   date        
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
def load_and_preprocess_image(image):
    """Loads the image, converts it to grayscale, and applies gamma correction."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = Gamma_correction(gray, 1.5)
    return gray, gray_image

def threshold_and_edge_detection(gray_image):
    """Applies thresholding and edge detection to preprocess the image for contour detection."""
    threshold = threshold_minimum(gray_image)
    mask = gray_image > threshold
    binary_image = np.zeros(gray_image.shape)
    binary_image[mask] = 255
    binary_image = binary_image.astype(np.uint8)
    edges = canny(binary_image, sigma=50, low_threshold=0.5, high_threshold=0.6)
    return Dilate(edges, 5)

def filter_reciept_contours(contours):
    """Filters contours to find a rectangular contour with 4 corners and a large area."""
    # rectangular_contours = []
    max_area = 0
    reciept_contour = None
    for contour in contours:
        # x, y, w, h = cv2.boundingRect(contour)
        # aspect_ratio = w / h
        # print(aspect_ratio)
        area = cv2.contourArea(contour)
        if area > max_area:
            peri = cv2.arcLength(contour, True)
            approx_polygon = cv2.approxPolyDP(contour, 0.015 * peri, True)
            print(len(approx_polygon))
            if area > max_area and len(approx_polygon) == 4:
                # rectangular_contours.append(approx_polygon)
                reciept_contour = approx_polygon
                max_area = area
    return reciept_contour

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
    gray, gray_image = load_and_preprocess_image(image) # Convert to grayscale and apply gamma correction
    edges = threshold_and_edge_detection(gray_image) # Apply thresholding and edge detection (Heba)
    # preProcessedImage, contours = apply_iterations(edges, edges)
    # reciept_contour = apply_iterations(gray_image, edges)
    
    connected_edges = cv2.dilate(edges, np.ones((60, 60)))
    show_images([connected_edges])
    contours, _ = cv2.findContours(fix_image_type(connected_edges), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    reciept_contour = filter_reciept_contours(contours)

    # contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # reciept_contour = filter_reciept_contours(contours)

    if len(reciept_contour) == 0:
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
    image = image.copy()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) * 255
    smoothed_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # binary_image = cv2.adaptiveThreshold(smoothed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # binary_image = np.bitwise_not(binary_image)

    binary_image = convert_to_binary(smoothed_image)

    dilation_image = dialation(binary_image, 3, 25)
    erosion_image = erosion(dilation_image, 3, 25)
    show_images([image, binary_image, dilation_image, erosion_image], ["Original", "Binary", "Dilation", "Erosion"])

    dilation_image = fix_image_type(dilation_image)

    # Get the contours of the image
    contours, _ = cv2.findContours(dilation_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours))

    # Filter the contours to get the 16 digits
    digits_contours = None
    largest_area = 0
    max_aspect_ratio = 0
    max_width = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        area = cv2.contourArea(contour)
        
        # ## Testing & Visualizing
        # print(aspect_ratio, area)
        # draw_contour = visualize_contours(image, [contour])
        # # show_images([draw_contour])
        # # save draw_contour with name of aspect_ratio
        # os.makedirs("aspect_ratios2", exist_ok=True)
        # io.imsave(f"aspect_ratios2/{aspect_ratio}.jpg", draw_contour)
        # ## Testing & Visualizing
        
        if width > max_width and aspect_ratio > max_aspect_ratio and area > largest_area and not is_between(aspect_ratio, 0.5, 1.5): #  and is_between(w, 20, 30) and is_between(h, 40, 50)
            digits_contours = contour
            largest_area = area
            ## Testing & Visualizing
            # draw_contour = visualize_contours(image, [contour])
            # show_images([draw_contour], [f"Aspect Ratio: {aspect_ratio}, Area: {area}"])

    if digits_contours is None:
        return image
    
    # ## Testing & Visualizing
    # show_images([visualize_contours(image, [digits_contours])])
    # ## Testing & Visualizing

    x, y, w, h = cv2.boundingRect(digits_contours)
    x, y, w, h = add_padding(x, y, w, h, padding_x=0, padding_y=4)

    return crop_image(image, x, y, w, h)
