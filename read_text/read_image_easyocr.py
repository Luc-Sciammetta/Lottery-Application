"""
This file contains the functionality for reading text from an image using EasyOCR. 
It defines a function `read_image` that takes the path to an image file and returns a list of the extracted text. 
"""

import easyocr
import numpy as np
import re

from PIL import Image

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ticket_classifiers.pretrained_ticket_classifier import load_model, predict_image

try:
    from read_text.perspective_correction import correct_image, read_image as open_image
except ImportError:
    from perspective_correction import correct_image, read_image as open_image

LOTTERY_CONFIGS = {
    "euromillions": {"main": 5, "special": 2, "special_label": "Lucky Stars"},
    "powerball":    {"main": 5, "special": 1, "special_label": "PB"},
    "megamillions": {"main": 5, "special": 1, "special_label": "MB"},
    "lottoamerica": {"main": 5, "special": 1, "special_label": "All Star Bonus"},
}

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


def parse_text(result, config):
    """ Parse the extracted text to find the relevant information.
    Args:
        result (list): A list of tuples containing the bounding box, text, and probability for each detected text.
        config (dict): The configuration for the lottery game.
    Returns:
        dict: A dictionary containing the parsed information.
    """
    possible_dates = []
    possible_weekdays = []
    filtered_text = [text for bbox, text, prob in result if prob > 0.25] # Filter out text with low probability

    #get the date of the lottery draw
    
    month_pattern = re.compile(r'\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC|JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s*\b', re.IGNORECASE)
    date_pattern = re.compile(r'\b(\d{1,2})\b') #tries to find any standalone numbers that could be a date, we will later filter these based on their proximity to month names and other special cases
    weekday_pattern = re.compile(r'\b(MON|TUE|WED|THU|FRI|SAT|SUN|MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY|SATURDAY|SUNDAY)\s*\b', re.IGNORECASE)
    month_I_pattern = re.compile(r'\b(JANI|FEBI|MARI|APRI|MAYI|JUNI|JULI|AUGI|SEPI|OCTI|NOVI|DECI)(\d)', re.IGNORECASE)
    month_O_pattern = re.compile(r'\b(JANO|FEBO|MARO|APRO|MAYO|JUNO|JULO|AUGO|SEPO|OCTO|NOVO|DECO)(\d)', re.IGNORECASE)
    month_Q_pattern = re.compile(r'\b(JANQ|FEBQ|MARQ|APRQ|MAYQ|JUNQ|JULQ|AUGQ|SEPQ|OCTQ|NOVQ|DECQ)(\d)', re.IGNORECASE)
    month_fused_with_date_pattern = re.compile(r'\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{1,2})\b',re.IGNORECASE)

    for text in filtered_text:
        special_case = ''
        month = None
        digit = None

        #look for any special cases
        month_I_match = month_I_pattern.search(text)
        month_O_match = month_O_pattern.search(text)
        month_Q_match = month_Q_pattern.search(text)
        month_fused_with_date_match = month_fused_with_date_pattern.search(text)

        if month_I_match:
            special_case = "I"
            month = month_I_match.group(1)[:-1]  # e.g. "JANI" -> "JAN"
            digit = month_I_match.group(2)       # e.g. "5"
        elif month_O_match:
            special_case = "O"
            month = month_O_match.group(1)[:-1]
            digit = month_O_match.group(2)
        elif month_Q_match:
            special_case = "Q"
            month = month_Q_match.group(1)[:-1]
            digit = month_Q_match.group(2)
        elif month_fused_with_date_pattern.search(text):
            special_case = "FUSED"
            month = month_fused_with_date_match.group(1)
            digit = month_fused_with_date_match.group(2)


        #look for perfect matches
        month_match = month_pattern.search(text)
        weekday_match = weekday_pattern.search(text)
        date_match = date_pattern.search(text)

        print(f"Checking text: '{text}' | Month: {month_match.group() if month_match else None}, Date: {date_match.group() if date_match else None}, Weekday: {weekday_match.group() if weekday_match else None}")
        print(f"SPECIAL: I={month_I_match.group() if month_I_match else None}, O={month_O_match.group() if month_O_match else None}, Q={month_Q_match.group() if month_Q_match else None}")

        #add the month and date to the possible dates list
        if month_match and date_match and special_case == '':
            possible_dates.append([month_match.group(), date_match.group()])
        elif special_case == "I" and digit:
            date = '1' + digit #adjust for special cases
            possible_dates.append([month, date])
        elif special_case == "O" and digit:
            date = '0' + digit
            possible_dates.append([month, date])
        elif special_case == "Q" and digit:
            date = '0' + digit
            possible_dates.append([month, date])
        elif special_case == "FUSED" and month and digit:
            possible_dates.append([month, digit])
        

        #if we match a weekday
        if weekday_match:
            possible_weekdays.append(weekday_match.group())


    return possible_dates


def read_image_text(image_path, game):
    corrected_image = correct_image(image_path) #correct the perspective of the image before reading the text
    if corrected_image is not None:
        result = get_text(corrected_image) #read the text from the corrected image
    else:
        print("[MAIN]: Could not correct the perspective of the image, reading text from the original image")
        image = open_image(image_path) #read the image using OpenCV
        result = get_text(image)
    
    
    for (bbox, text, prob) in result:
        print(f'Text: {text}, Probability: {prob}')

    result = parse_text(result, LOTTERY_CONFIGS.get(game)) #parse the extracted text to find the relevant information

    return result


if __name__ == "__main__":
    image_path = "images/powerball/img9.jpg"

    model = load_model("ticket_classifier_models/pt_88.52_88.89_model_weights.pth") #load a pre-trained model
    game = predict_image(model, image_path) #path to a test image
    
    print(f"Reading text from image: {image_path}")
    result = read_image_text(image_path, game)
    print("Parsed Result:")
    print(result)

    # image_path = "images/megamillions/IMG_3451.jpeg"
    # print(f"Reading text from image: {image_path}")
    # result = read_image_text(image_path)
    # image_path = "images/lottoamerica/img8.jpg"
    # print(f"Reading text from image: {image_path}")
    # result = read_image_text(image_path)
    # image_path = "images/euromillions/img3.jpg"
    # print(f"Reading text from image: {image_path}")
    # result = read_image_text(image_path)
    # image_path = "images/powerball/img14.jpg"
    # print(f"Reading text from image: {image_path}")
    # result = read_image_text(image_path)
    
    # parsed_data = parse_text(result)
    # print(parsed_data)