import cv2
import pytesseract
import numpy as np
from PIL import Image

def process_receipt_image(image):
    """
    Process the receipt image and extract data using OCR.
    Args:
        image: Uploaded receipt image.
    Returns:
        Dictionary with extracted data.
    """
    try:
        # Convert image to OpenCV format
        image = load_image(image)


        reciept = find_reciept("../imgs/10.jpg")

        # Preprocess the image (grayscale, thresholding)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(reciept, 128, 255, cv2.THRESH_BINARY)
        
        # Perform OCR
        raw_text = pytesseract.image_to_string(binary)
        #raw_text = "Date: 12/12/2020\nTotal: $100.00"

        # Parse the OCR output (example parsing logic)
        lines = raw_text.split("\n")
        data = {}
        for line in lines:
            if "Date" in line:
                data["Date"] = line.split(":")[-1].strip()
            elif "Total" in line or "$" in line:
                data["Total Amount"] = line.split(":")[-1].strip()

        return data
    except Exception as e:
        print(f"Error: {e}")
        return None


#
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
