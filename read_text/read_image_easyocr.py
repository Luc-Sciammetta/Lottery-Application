"""
This file contains the functionality for reading text from an image using EasyOCR. 
It defines a function `read_image` that takes the path to an image file and returns a list of the extracted text. 
"""

import easyocr
import numpy as np

from PIL import Image

from read_text.perspective_correction import correct_image, read_image as open_image

def get_text(image):
    """ Read text from an image using EasyOCR.
    Args:
        image (numpy array): The image to read text from.
    Returns:
        list: A list of tuples containing the bounding box, text, and probability for each detected text.
    """

    img = np.array(Image.fromarray(image).convert('L')) #convert to grayscale for better OCR performance and then convert to numpy array for easyOCR

    reader = easyocr.Reader(['en', 'es', 'fr', 'de']) # Initialize the EasyOCR reader with the desired languages
    result = reader.readtext(img) # Read the text from the image

    # for (bbox, text, prob) in result:
    #     print(f'Text: {text}, Probability: {prob}')

    return result


def parse_text(result):
    """ Parse the extracted text to find the relevant information.
    Args:
        result (list): A list of tuples containing the bounding box, text, and probability for each detected text.
    Returns:
        dict: A dictionary containing the parsed information.
    """
    potential_numbers = []
    potential_dates = []
    
    for (bbox, text, prob) in result:
        splitted_text = text.split(" ")
        for word in splitted_text:
            if word.isdigit() and len(word) == 2: #assuming that the numbers on the ticket are always 2 digits long
                pass
        pass

    # return parsed_data


def read_image_text(image_path):
    corrected_image = correct_image(image_path) #correct the perspective of the image before reading the text
    if corrected_image is not None:
        result = get_text(corrected_image) #read the text from the corrected image
    else:
        print("[MAIN]: Could not correct the perspective of the image, reading text from the original image")
        image = open_image(image_path) #read the image using OpenCV
        result = get_text(image)
    
    
    for (bbox, text, prob) in result:
        print(f'Text: {text}, Probability: {prob}')

    return result
    

if __name__ == "__main__":
    image_path = "images/powerball/image - 1.jpeg"

    result = read_image_text(image_path)
    
    # parsed_data = parse_text(result)
    # print(parsed_data)