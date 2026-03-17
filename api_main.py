import pprint
import pandas as pd
import datetime

import abstract_funcs
from read_text.read_image_easyocr import read_image_text

#if using a non-pre-trained model, use these imports
# from ticket_classifiers.ticket_classifier import load_model
# from ticket_classifiers.test_ticket_classifier import predict_image

#if using a pre-trained model, use these imports
from ticket_classifiers.pretrained_ticket_classifier import load_model, predict_image

df = pd.DataFrame()
HMS = False #we always want to ignore time in the comparison for this method

months = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12 } 

def clean_numbers(draw_numbers):
    """
    Cleans the extracted numbers by stripping whitespace and converting to integers.
    Args:
        draw_numbers (list): A list of lists containing the extracted numbers.
    Returns:        
        list: A list of lists containing the cleaned numbers.
    """
    clean_numbers = []
    for group in draw_numbers:
        clean_group = []
        for number in group:
            if isinstance(number, str):
                number = number.strip()
                if number.isdigit():
                    clean_group.append(int(number))
        clean_numbers.append(clean_group)
    return clean_numbers


def convert_date(dates):
    """Converts a list of dates into the format YYYY-MM-DD
    Args:
        dates (list): A list of dates
    Returns:
        list: A list of converted dates in the format YYYY-MM-DD
    """
    if len(dates) == 0:
        return None

    list_of_dates = []
    for i in range(len(dates)):
        group = dates[i]
        month = group[0].upper().strip()
        day = group[1]
        if day.isdigit():
            day = int(day)
        else:
            raise ValueError(f"Invalid day: {day}")

        if month in months:
            month = months[month]
        else:
            raise ValueError(f"Invalid month: {month}")

        current_year = datetime.date.today().year
        if int(month) > datetime.date.today().month:
            date = f"{current_year-1}-{month}-{day:02d}"
        else:
            date = f"{current_year}-{month}-{day:02d}"
        
        list_of_dates.append(date)
    return list_of_dates

    
def main():
    image_path = "images/powerball/image - 1.jpeg"

    model = load_model("ticket_classifier_models/pt_88.52_88.89_model_weights.pth") #load a pre-trained model
    ticket_type = predict_image(model, image_path) #path to a test image

    dates, weekdays, draw_numbers, draw_special = read_image_text(image_path, ticket_type) #path to a test image and the predicted ticket type

    print("BEFORE CLEANING Draw Numbers:", draw_numbers)
    print("BEFORE CLEANING Draw Special:", draw_special)
    print("BEFORE CONVERTING Dates:", dates)
    
    draw_numbers = clean_numbers(draw_numbers)
    draw_special = clean_numbers(draw_special)

    converted_dates = convert_date(dates)

    print(f"Predicted Ticket Type: {ticket_type}")
    print("Extracted Dates:", converted_dates)
    print("Extracted Weekdays:", weekdays)
    print("Extracted Draw Numbers:", draw_numbers)
    print("Extracted Draw Special:", draw_special)

    global df
    df = abstract_funcs.convert_to_pd(f"lottery_data/{ticket_type}", df)

    for i in range(len(draw_numbers)):
        if converted_dates is None:
            numbers = draw_numbers[i]
            specials = draw_special[i]
            pprint.pprint(abstract_funcs.check_for_win(ticket_type, numbers, specials, df))
        else:
            numbers = draw_numbers[i]
            specials = draw_special[i]
            try:
                pprint.pprint(abstract_funcs.check_for_win(ticket_type, numbers, specials, df, converted_dates))
            except ValueError as e:
                print(f"Error checking for win: {e}")

main()