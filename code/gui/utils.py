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
        image = Image.open(image)
        image = np.array(image)

        # Preprocess the image (grayscale, thresholding)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

        # Perform OCR
        raw_text = pytesseract.image_to_string(binary)

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

process_receipt_image("imgs/1.jpg")