"""
This file contains the functionality for reading text from an image using EasyOCR. 
It defines a function `read_image` that takes the path to an image file and returns a list of the extracted text. 
"""

from logging import config

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
    "euromillions": {"main": 5, "special": 2, "special_labels": ["Lucky Stars", "--", '-', '++']},
    "powerball":    {"main": 5, "special": 1, "special_labels": ['PB', "EP", "QP", "OP", "-", "PWR"]},
    "megamillions": {"main": 5, "special": 1, "special_labels": ['MB', "EP", "QP", "OP", "AP"]},
    "lottoamerica": {"main": 5, "special": 1, "special_labels": ['All Star Bonus', "EP", "QP", "OP", "SB"]},
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
    filtered_text = [text for bbox, text, prob in result if prob > 0.15] # Filter out text with low probability

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

        # print(f"Checking text: '{text}' | Month: {month_match.group() if month_match else None}, Date: {date_match.group() if date_match else None}, Weekday: {weekday_match.group() if weekday_match else None}")
        # print(f"SPECIAL: I={month_I_match.group() if month_I_match else None}, O={month_O_match.group() if month_O_match else None}, Q={month_Q_match.group() if month_Q_match else None}")

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

    print("Possible Dates:", possible_dates)
    print("Possible Weekdays:", possible_weekdays)

    paren_number_pattern = re.compile(r'^\(\d+\)$')  # matches (22), (5), etc.
    row_label_pattern = re.compile(r'^([A-J])\.?\s?(\d+)?$', re.IGNORECASE)

    draw_numbers = []
    draw_special = []

    rows = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J'}
    draw_row = 0
    current_text_index = 0

    while True:
        current_text = filtered_text[current_text_index]
        match = row_label_pattern.match(current_text) #match to see if the text is a row label like "A", "B", etc. with an optional number after it (e.g. "A.1", "B2", etc.)
        if match: #we have a match
            letter = match.group(1).upper() #get the letter
            
            #find the right matched row (so A = 0, B = 1, etc.)
            matched_row = None 
            for k, v in rows.items():
                if v == letter:
                    matched_row = k
                    break

            if matched_row is not None and matched_row >= draw_row:
                #look ahead to see if there are numbers afterward, indicating we have found a valid row label.
                lookahead_found = False
                for i in range(1, 4): #look ahead 3 rows
                    if current_text_index + i >= len(filtered_text):
                        break
                    peek_tokens = filtered_text[current_text_index + i].split()
                    for peek_token in peek_tokens:
                        if peek_token.isdigit() and len(peek_token) <= 2:
                            #we found a number in the lookahead, this is a good sign that we have found a valid row label 
                            lookahead_found = True
                            break
                    if lookahead_found:
                        break

                if lookahead_found:
                    # fill skipped rows with empty lists
                    while draw_row < matched_row:
                        draw_numbers.append([])
                        draw_special.append([])
                        draw_row += 1

                    trailing = match.group(2) if match.lastindex >= 2 else None
                    found_numbers = [trailing] if trailing else []
                    found_special = []
                    ahead_row_index = 0
                    special_label_token = False
                    special_label_token_index = -1
                    special_label_tokens = None

                    #find main numbers (and grab special if possible)
                    while len(found_numbers) < config['main']:
                        tokens = filtered_text[current_text_index + ahead_row_index].split()
                        for token in tokens:
                            if token.isdigit() and len(token) <= 2:
                                if len(found_numbers) < config['main']: #add only if we still have room in the found_numbers list, otherwise this digit must be the special number
                                    found_numbers.append(token)
                                else:
                                    # found_numbers is full, this digit must be the special number
                                    found_special.append(token)
                                    break
                            elif paren_number_pattern.match(token)  and len(token) <= 4: #2 for the number and 2 for the ()
                                found_special.append(token.strip('()'))
                                break
                            elif token in config['special_labels']:
                                special_label_token = True
                                special_label_token_index = tokens.index(token)
                                special_label_tokens = tokens
                                break
                            else:
                                break
                        ahead_row_index += 1
                        if current_text_index + ahead_row_index >= len(filtered_text):
                            break
                            
                    draw_numbers.append(found_numbers)                
                    print(f"[DEBUG] After main number loop: ahead_row_index={ahead_row_index}, found_numbers={found_numbers}, found_special={found_special}")


                    #find special numbers (only if not already found)
                    if len(found_special) < config['special']:
                        exit = False
                        if special_label_token: #we have seen the special label, so we know there will be a special number soon.
                            if special_label_token_index + 1 < len(special_label_tokens):
                                # numbers are to the right of the label on the same row
                                for token in special_label_tokens[special_label_token_index + 1:]:
                                    if token.isdigit() and len(token) <= 2:
                                        found_special.append(token)
                                    else:
                                        break
                            else:
                                # numbers are on the next row(s)
                                while len(found_special) < config['special'] and not exit:
                                    if current_text_index + ahead_row_index >= len(filtered_text):
                                        break
                                    tokens = filtered_text[current_text_index + ahead_row_index].split()
                                    for token in tokens:
                                        if token.isdigit() and len(token) <= 2:
                                            found_special.append(token)
                                        else: #we have found text or a number that is not a special number
                                            exit = True
                                            break
                                    ahead_row_index += 1
                        else:
                            # no label found, scan up to 3 rows below for a special label or parenthesized number
                            buffer = 3
                            while buffer > 0 and not exit:
                                if current_text_index + ahead_row_index >= len(filtered_text):
                                    break
                                tokens = filtered_text[current_text_index + ahead_row_index].split()
                                print(f"[DEBUG BUFFER] Scanning row: {filtered_text[current_text_index + ahead_row_index]} | ahead_row_index={ahead_row_index} | buffer={buffer}")
                                for token in tokens:
                                    if paren_number_pattern.match(token): 
                                        # parenthesized number — grab it directly
                                        found_special.append(token.strip('()'))
                                        exit = True
                                        break
                                    elif token in config['special_labels']: #we found a special label
                                        # found the label — now grab the number after it
                                        special_label_token = True
                                        special_label_token_index = tokens.index(token)
                                        special_label_tokens = tokens
                                        exit = True
                                        break
                                    else:
                                        exit = True
                                        break
                                if not exit:
                                    ahead_row_index += 1
                                buffer -= 1

                            # if we found a label in the buffer scan, find the special number the same way as before
                            if special_label_token and len(found_special) < config['special']:
                                if special_label_token_index + 1 < len(special_label_tokens):
                                    for token in special_label_tokens[special_label_token_index + 1:]:
                                        if token.isdigit() and len(token) <= 2:
                                            found_special.append(token)
                                        else:
                                            break
                                else:
                                    # number is on the next row
                                    label_exit = False
                                    while len(found_special) < config['special'] and not label_exit:
                                        if current_text_index + ahead_row_index >= len(filtered_text):
                                            break
                                        tokens = filtered_text[current_text_index + ahead_row_index].split()
                                        for token in tokens:
                                            if token.isdigit() and len(token) <= 2:
                                                found_special.append(token)
                                            else:
                                                label_exit = True
                                                break
                                        ahead_row_index += 1

                    #for some lottery tickets, the OCR doesnt pick up on the special label, but the special numbers are right after
                    #the main numbers, so if this happens, we will grab any numbers that come right after the main numbers as the special numbers until we hit a non-digit or we have enough special numbers. 
                    if len(found_special) < config['special']:
                        fallback_exit = False
                        while len(found_special) < config['special'] and not fallback_exit:
                            if current_text_index + ahead_row_index >= len(filtered_text):
                                break
                            tokens = filtered_text[current_text_index + ahead_row_index].split()
                            for token in tokens:
                                if token.isdigit() and len(token) <= 2:
                                    found_special.append(token)
                                    if len(found_special) >= config['special']:
                                        break
                                else:
                                    fallback_exit = True
                                    break
                            ahead_row_index += 1

                    draw_special.append(found_special)
                    draw_row += 1

        current_text_index += 1
        if current_text_index >= len(filtered_text) or draw_row >= len(rows):
            break


    print("Draw Numbers after row label parsing:", draw_numbers)
    print("Draw Special after row label parsing:", draw_special)


    #now there may be some draw numbers that didnt have a row label which we might still be able to find,
    #so we will look through the text again and try to find any numbers that are not already in our
    #draw numbers list that look like the way the draw numbers should beand add them to the final lists.
    i = 0
    while i < len(filtered_text):
        print("[DEBUG] Checking text for unlabeled numbers:", filtered_text[i])
        tokens = filtered_text[i].split()
        candidate_numbers = []
        candidate_special = []
        special_label_token = False
        special_label_token_index = -1
        special_label_tokens = None
        ahead = 0
        done = False #keeps track to see if we have enough info to make a group of numbers

        while (len(candidate_numbers) < config['main'] or (special_label_token and len(candidate_special) < config['special'])) and not done:
            if i + ahead >= len(filtered_text):
                break
            tokens = filtered_text[i + ahead].split()
            print(f"[DEBUG] Lookahead text: {filtered_text[i + ahead]} | Tokens: {tokens}")
            for token in tokens:
                print(f"[DEBUG] Checking token: {token} | Candidate Numbers: {candidate_numbers} | Candidate Special: {candidate_special} | Special Label Token: {special_label_token}")
                if token.isdigit() and len(token) <= 2:
                    if special_label_token: #then this digit is the special number
                        candidate_special.append(token)
                        done = True
                        break
                    elif len(candidate_numbers) < config['main']:
                        candidate_numbers.append(token)
                    else:
                        candidate_numbers = [] #we have an extra main number, so something isnt right here, so we stop
                        done = True
                        break
                elif paren_number_pattern.match(token) and len(token) <= 4:
                    candidate_special.append(token.strip('()'))
                    done = True
                    break
                elif token in config['special_labels']:
                    special_label_token = True
                    special_label_token_index = tokens.index(token)
                    special_label_tokens = tokens
                else: #we have encountered a token that is not a number or a special label, so we can stop looking
                    candidate_numbers = []
                    done = True
                    break
            ahead += 1
        
        #now check to see if we have already found this group of numbers
        if len(candidate_numbers) == config['main']:
            already_found = any(candidate_numbers == entry for entry in draw_numbers)
            if not already_found:
                print(f"[DEBUG] Found candidate numbers: {candidate_numbers} | Candidate special: {candidate_special}")
                draw_numbers.append(candidate_numbers)
                draw_special.append(candidate_special if len(candidate_special) > 0 else [])
        i += 1

    print("Draw Numbers after lookahead parsing:", draw_numbers)
    print("Draw Special after lookahead parsing:", draw_special)

    #for some reason, we might get duplicates/false draw numbers from the above logic, so we try to 
    #find those and then remove them here
    to_remove = set() #stores the indicies of draw numbers that we want to remove
    for i in range(len(draw_numbers)):
        if i in to_remove:
            continue #we have already marked this index for removal, so skip it
        
        for j in range(i + 1, len(draw_numbers)):
            if j in to_remove:
                continue #we have already marked this index for removal, so skip it
            
            entry_a = draw_numbers[i] #one row of the draw numbers
            entry_b = draw_numbers[j] #another row of the draw numbers that is to be compared to entry_a

            if len(entry_a) == 0 or len(entry_b) == 0:
                continue #skip empty entries

            # count how many numbers appear in both entries
            matches = 0
            for num in entry_a:
                if num in entry_b:
                    matches += 1

            if matches >= config['main'] - 1: 
                to_remove.add(j) #mark for removal

    #remake the lists without the marked entries
    draw_numbers = [entry for idx, entry in enumerate(draw_numbers) if idx not in to_remove]
    draw_special = [entry for idx, entry in enumerate(draw_special) if idx not in to_remove]

    print("Draw Numbers after duplicate removal:", draw_numbers)
    print("Draw Special after duplicate removal:", draw_special)
    
    return possible_dates, possible_weekdays, draw_numbers, draw_special


def read_image_text(image_path, game):
    corrected_image = correct_image(image_path) #correct the perspective of the image before reading the text
    if corrected_image is not None:
        result = get_text(corrected_image) #read the text from the corrected image
    else:
        print("[MAIN]: Could not correct the perspective of the image, reading text from the original image")
        image = open_image(image_path) #read the image using OpenCV
        result = get_text(image)
    
    counter = 0
    for (bbox, text, prob) in result:
        print(f'Counter: {counter}, Text: {text}, Probability: {prob}')
        counter += 1

    result = parse_text(result, LOTTERY_CONFIGS.get(game)) #parse the extracted text to find the relevant information

    return result


if __name__ == "__main__":
    image_path = "images/powerball/img1.jpg"
    # image_path = "images/lottoamerica/img3.jpg"


    model = load_model("ticket_classifier_models/pt_88.52_88.89_model_weights.pth") #load a pre-trained model
    game = predict_image(model, image_path) #path to a test image
    
    print(f"Reading text from image: {image_path}")
    dates, weekdays, draw_numbers, draw_special = read_image_text(image_path, game)
    print("Parsed Result:")
    print("Possible Dates:", dates)
    print("Possible Weekdays:", weekdays)
    print("Draw Numbers:", draw_numbers)
    print("Special Numbers:", draw_special)
    

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