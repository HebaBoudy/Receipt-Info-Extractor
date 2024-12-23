import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.measure import find_contours
from scipy.ndimage import binary_erosion, binary_closing, binary_opening, binary_dilation
from common_functions import * 
from image_transformation import * 
import heapq
PITESSERACT = False
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
    receiptGrey = cv2.cvtColor(receipt, cv2.COLOR_BGR2GRAY)
    return receipt, receiptGrey
    

def segment_receipt_by_colors(image: np.ndarray):
    """
    Segment the receipt image by colors using K-Means clustering.
    Args:
        image: Extracted receipt image.
    Returns:
        Segmented receipt image.
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

    return segmented_image, opened

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



def find_digits(image: np.ndarray) -> np.ndarray:
    """Main functionality is to detect the 16 digits from the receipt."""
    height, width, _ = image.shape
    image = image.copy()
    image = gamma_correction(image, 1.5)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) * 255
    smoothed_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    

    # TODO: Try apply contrast enhancement then use minimum thresholding instead of Otsu
    binary_image = convert_to_binary(smoothed_image)

    dilation_image = dialation(binary_image, 3, 25)
    erosion_image = erosion(dilation_image, 3, 25)
    # show_images([image, gray_image, binary_image, dilation_image, erosion_image], ["Original", "Gray", "Binary", "Dilation", "Erosion"])

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
        
        ## Testing & Visualizing
        # print(aspect_ratio, area)
        # draw_contour = visualize_contours(image, [contour])
        # show_images([draw_contour], [f"{w}, {h}"])
        # save draw_contour with name of aspect_ratio
        # os.makedirs("aspect_ratios2", exist_ok=True)
        # io.imsave(f"aspect_ratios2/{aspect_ratio}.jpg", draw_contour)
        ## Testing & Visualizing

        # is_between(w, 0.7 * width, 0.8 * width)
        # w > max_width
        # and aspect_ratio > max_aspect_ratio        
        if is_between(w, 0.6 * width, 0.8 * width) and area > largest_area and not is_between(aspect_ratio, 0.5, 1.5): #  and is_between(w, 20, 30) and is_between(h, 40, 50)
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

def split_digits(digits: np.ndarray):
    if digits is None or digits.size == 0:
        raise ValueError("Input image is empty or not loaded correctly.")
    
    # Convert to grayscale
    gray = cv2.cvtColor(digits, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create a binary image
    binary = convert_to_binary(gray)
    binary = np.bitwise_not(binary)

    max_iter = 6
    vertical_erosion = 2
    for i in range(max_iter):
        if i != 0:
            eroded = erosion(binary, vertical_erosion, 1)
        else:
            eroded = binary

        show_images([eroded], ["Binary"])

        # Find contours
        contours, _ = cv2.findContours(fix_image_type(eroded), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        digits_splited = []
        max_height = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x, y, w, h = add_padding(x, y, w, h, padding_x=4, padding_y=5)
            digits_splited.append((crop_image(digits, x, y, w, h), x))
            max_height = max(max_height, h) # Save it to remove small contours

        digits_splited.sort(key=lambda x: x[1])
        digits_splited = [x[0] for x in digits_splited if x[0].shape[0] > 0.5 * max_height]
        if len(digits_splited) == 16:
            break
        vertical_erosion += 1

    return digits_splited


def match_template(image, templates_folder):

    best_match = None
    highest_corr = -1
    
    for digit in range(10):  # Loop through 0-9 folders
        digit_folder = os.path.join(templates_folder, str(digit))
        
        # Check if the digit folder exists
        if not os.path.exists(digit_folder):
            continue
        
        # Loop through all images inside the current digit folder
        for template_filename in os.listdir(digit_folder):
            template_path = os.path.join(digit_folder, template_filename)
            template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            
            # Ensure the template and image are the same size
            if image.shape == template_img.shape:
                # Perform template matching using normalized cross-correlation
                result = cv2.matchTemplate(image, template_img, cv2.TM_CCOEFF_NORMED)
                corr_value = result[0][0]  # Single value since image size == template size

                # Track the best match
                if corr_value > highest_corr:
                    highest_corr = corr_value
                    best_match = str(digit)
    print("finished match template" ,best_match)
    
    return best_match

def template_preprocess_image(img, dimensions):
    img_resized = cv2.resize(img, dimensions)
    
    # Convert RGBA to RGB if necessary
    if len(img_resized.shape) == 3 and img_resized.shape[2] == 4:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGBA2RGB)
    
    # Convert to grayscale
    #check if the image is already in grayscale
    if len(img_resized.shape) == 3:
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_resized
    #show_images([img_gray], ["Gray Image"])

    #now make it binary
    img_binary = np.zeros(img_gray.shape)
    img_binary[img_gray > 0.5] = 255
    #img_binary *= 255
    img_binary = img_binary.astype(np.uint8)
    #show_images([img_binary], ["Binary Image"])
    # print(img_gray.max())
    # print("finished preprocess image")
    return img_binary


def post_template_matching(digits):
    # Loop through each letter image
    predicted_digits = []
    templates_folder = "code/templates2"  # Path to your templates2 folder
    print(os.path.exists(templates_folder))
    #show_images(letters, ["Letter Image"])

    for digit_img in digits:
        #show_images([letter_img])  # Display the current letter

        # Preprocess the image (ensure it matches template dimensions)
        processed_img = template_preprocess_image(digit_img, (185,386))
        # show_images([digit_img, processed_img], ["Original", "Processed Image"])

        # Perform template matching
        predicted_digit = match_template(processed_img, templates_folder)
        predicted_digits.append(predicted_digit)

    if len(predicted_digits) == 16 and all([digit is not None for digit in predicted_digits]):
        # print("Predicted Digits:", "".join(predicted_digits))

        predicted_digits = "".join(predicted_digits)
        predicted_digits = ' '.join([predicted_digits[i:i+4] for i in range(0, len(predicted_digits), 4)])
        return "".join(predicted_digits)
    else:
        print("Error: Could not detect all digits.")
        return None

def process_contours(img,min_contour_size = 100):
    global contours, queue, existing_regions, letters

    # Find contours of the digits
    contours = find_contours(img, 0.8)
    print("Number of contours found:", len(contours))

    # Keep only the big contours
    contours = [contour for contour in contours if contour.shape[0] > min_contour_size]
    # print("Number of contours after filtering:", len(contours))

    # fig, ax = plt.subplots()
    # ax.imshow(img, cmap=plt.cm.gray)
    # for contour in contours:
    #     ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')
    # ax.axis('image')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # plt.show()

    queue = []
    existing_regions = []  # To store existing bounding boxes

    # Function to check intersection between bounding boxes
    def intersects(new_box, existing_boxes):
        new_x1, new_y1, new_x2, new_y2 = new_box
        for box in existing_boxes:
            x1, y1, x2, y2 = box
            if not (new_x2 < x1 or new_x1 > x2 or new_y2 < y1 or new_y1 > y2):
                return True  # Intersection found
        return False

    # Extract bounding boxes for contours
    for contour in contours:
        maxY = contour[:, 0].max()
        maxX = contour[:, 1].max()
        minY = contour[:, 0].min()
        minX = contour[:, 1].min()

        
        
        new_box = (minX, minY, maxX, maxY)
        if not intersects(new_box, existing_regions):
            # Push regions to a heap for sorting by X-coordinate
            heapq.heappush(queue, (minX, img[int(minY):int(maxY), int(minX):int(maxX)]))
            existing_regions.append(new_box)

    # Extract individual digit images
    letters = []
    while len(queue) > 0:
        img_digit = heapq.heappop(queue)[1]  # Extract the digit region
        img_digit = img_digit.astype(np.uint8)
        img_digit = img_digit.max() - img_digit  # Invert colors if necessary
        letters.append(img_digit)  # Append the digit image to the list
    return letters

def split_large_image(images,val = 1.6):
    # Step 1: Get column (width) sizes for all images
    col_sizes = [img.shape[1] for img in images]
    avg_col_size = sum(col_sizes) / len(col_sizes)

    # Step 2: Identify the image with double the column size
    for i, img in enumerate(images):
        if img.shape[1] >= val * avg_col_size:  # Flexibility with threshold (e.g., 1.8 instead of 2)
            # Step 3: Split the image vertically into two halves
            mid_col = img.shape[1] // 2
            left_half = img[:, :mid_col]  # Left part
            right_half = img[:, mid_col:]  # Right part

            # Step 4: Replace the large image with the two halves
            images[i] = left_half
            images.insert(i + 1, right_half)
            break  # Stop after splitting the first large image

    return images

def find_digits_basel(receiptGrey):
    threshold = 0.5
    receiptBinary = receiptGrey > threshold
    receiptBinaryInverted = np.invert(receiptBinary)
    
    # 1024 650 , numpy ndarr
    receiptBinaryInverted = binary_closing(receiptBinaryInverted,np.ones((2,2)), iterations=1) #problem 2
    
    receiptBinaryCropped = receiptBinaryInverted[400:750, 0:]
    
    binarydilated = binary_dilation(receiptBinaryCropped,np.ones((5,20)),iterations=2) 
   
    binarydilated = np.uint8(binarydilated * 255)
    

    # Find contours in the binary-dilated image
    contours, _ = cv2.findContours(binarydilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    max_area = 0
    max_contour = None

    # Iterate through the contours and calculate the area
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    # If a valid contour is found, calculate its bounding box and crop the region
    cropped_receipt = None
    if max_contour is not None:
        x, y, w, h = cv2.boundingRect(max_contour)
        cropped_receipt = receiptBinaryCropped[y:y+h, x:x+w] 

    img = cropped_receipt
    # process_contours(cropped_receipt)

    min_contour_size = 100
    while True:
        process_contours(img,min_contour_size) # here 
        print("Current min_contour_size:", min_contour_size)
        print("Number of letters:", len(letters))
        if len(letters) >= 16 or min_contour_size <= 40:
            break
        min_contour_size -= 5  # Reduce the contour size threshold incrementally
    for i in range(8):
        if len(letters) < 16:
            split_images = split_large_image(letters,1.5) 


    return  post_template_matching(letters)
def findDigitsInDilated(binarydilated, receiptBinaryCropped, numberOfContours):
    """Finds and shows the cropped areas of the largest contour, sorted by vertical position."""
    # Convert binarydilated to uint8 type if it's a boolean array
    
    binarydilated = np.uint8(binarydilated * 255)

    # Find contours in the binary-dilated image
    contours, _ = cv2.findContours(binarydilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    max_area = 0
    max_contour = None
    
    # Iterate through the contours and calculate the area
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour
    
    # If a valid contour is found, calculate its bounding box and crop the region
    if max_contour is not None:
        x, y, w, h = cv2.boundingRect(max_contour)
        cropped_receipt = receiptBinaryCropped[y:y+h, x:x+w]
        return cropped_receipt
    else:
        return None

def match_template_with_symbols(image, templates_folder):
    best_match = None
    highest_corr = -1
    
    # Define folder names and corresponding symbols
    symbol_mapping = {
        "0": "0", "1": "1", "2": "2", "3": "3", "4": "4",
        "5": "5", "6": "6", "7": "7", "8": "8", "9": "9"
        #"E": "E", "G": "G", "P": "P", "aa": "."
    }
    
    for folder_name, symbol in symbol_mapping.items():  # Loop through folders
        folder_path = os.path.join(templates_folder, folder_name)
        
        # Check if the folder exists
        if not os.path.exists(folder_path):
            continue
        
        # Loop through all images inside the current folder
        for template_filename in os.listdir(folder_path):
            template_path = os.path.join(folder_path, template_filename)
            template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            
            # Ensure the template and image are the same size
            if image.shape == template_img.shape:
                # Perform template matching using normalized cross-correlation
                result = cv2.matchTemplate(image, template_img, cv2.TM_CCOEFF_NORMED)
                corr_value = result[0][0]  # Single value since image size == template size

                # Track the best match
                if corr_value > highest_corr:
                    highest_corr = corr_value
                    best_match = symbol  # Use the symbol for this folder
    
    return best_match

def get_price(receiptGrey):
    threshold = 0.5
    receiptBinary = receiptGrey > threshold
    receiptBinaryInverted = np.invert(receiptBinary)
    
    # 1024 650 , numpy ndarr
    receiptBinaryInverted = binary_closing(receiptBinaryInverted,np.ones((2,2)), iterations=1) #problem 2
    bottomPart2 = receiptBinaryInverted[int(3*receiptBinaryInverted.shape[0]/4 - 70):int(receiptBinaryInverted.shape[0]-100),
                                     0:int(receiptBinaryInverted.shape[1]/2)].copy()
    
    binarydilated2 = binary_dilation(bottomPart2,np.ones((3,20)),iterations=2)
    croppedNumbers = findDigitsInDilated(binarydilated2,bottomPart2,4)
    img = croppedNumbers
    letters = []
    min_contour_size = 70
    while True:
        img = croppedNumbers
        letters = process_contours(img,min_contour_size)
        print("Current min_contour_size:", min_contour_size)
        print("Number of letters:", len(letters))
        if len(letters) >= 7 or min_contour_size <= 40:
            break
        min_contour_size -= 5  # Red

    # Loop through each letter image
    predictedPrice = []
    templates_folder = "code/templates2"
    letters = letters[:4]
    for letter_img in letters:
        # Preprocess the image (ensure it matches template dimensions)
        processed_img = template_preprocess_image(letter_img, (185, 386))

        # Perform template matching
        predicted_symbol = match_template_with_symbols(processed_img, templates_folder)
        predictedPrice.append(predicted_symbol)
    #The insert a dot before the last 2 digits
    predictedPrice.insert(-2, ".")
    #apped EGP to the end
    predictedPrice.append("EGP")
    # Display the result 
    print("predicted price",predictedPrice)
    if None not in predictedPrice :
        print("Predicted Symbols:", "".join(predictedPrice))
        return "".join(predictedPrice)
    else : 
        return None
    
