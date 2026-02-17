"""
This file contains the functionality for reading text from an image using Tesseract OCR. 
It defines a function `read_image` that takes the path to an image file and returns the extracted text in uppercase. 
However, from initial usage, it seems as if Tesseract is not reliably able to read the text from the image. 
"""

import pytesseract

from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

def get_text(image_path, timeout=2):
    """ Read text from an image using Tesseract OCR.
    Args:
        image_path (str): The path to the image file.
        timeout (int): Maximum time to wait for Tesseract to process the image.
    Returns:
        str: The text extracted from the image in uppercase.
    """
    # Open the image using PIL
    # so that we can convert to grayscale
    img = Image.open(image_path)

    # Convert to grayscale ('L' mode)
    img_gray = img.convert('L')

    try:
        text = pytesseract.image_to_string(img_gray, timeout=timeout)
    except RuntimeError as timeout_error:
        print(f"Timeout Reading image {image_path}")
        return ""

    text = text.strip().upper()

    return text

# # Get bounding box estimates
# print(pytesseract.image_to_boxes(Image.open('test.png')))

# # Get verbose data including boxes, confidences, line and page numbers
# print(pytesseract.image_to_data(Image.open('test.png')))

# # Get information about orientation and script detection
# print(pytesseract.image_to_osd(Image.open('test.png')))

# # getting multiple types of output with one call to save compute time
# # currently supports mix and match of the following: txt, pdf, hocr, box, tsv
# text, boxes = pytesseract.run_and_get_multiple_output('test.png', extensions=['txt', 'box'])