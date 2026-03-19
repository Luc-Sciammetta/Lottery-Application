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
        day = group[1].strip()
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


def change_details(dates, draw_numbers, draw_special):
    print("\nWhich detail would you like to change?")
    print("d=ates, n=draw numbers, s=special numbers")
    print("Then type in the index of the detail you want to change, and the new value")
    print("EX: 'd 0 2024-01-01' to change the first date to January 1st, 2024")
    print("EX: 'n 0 2 15' to change the third number in the first group of draw numbers to 15")
    user_input = input().strip().split()
    detail_type = user_input[0]
    index = int(user_input[1])
    if detail_type == 'd':
        new_date = user_input[2]
        if len(new_date.split('-')) == 3:
            if len(dates) < index + 1:
                dates.append(new_date)
            dates[index] = new_date
            print(dates)
        else:
            print("\nInvalid date format. Please use YYYY-MM-DD.")
    elif detail_type == 'n':  
        new_number = int(user_input[3])
        print("NEW NUMBER:", new_number)

        draw_numbers[index][int(user_input[2])] = new_number
    elif detail_type == 's':
        new_number = int(user_input[3])
        if len(draw_special) < index + 1:
            draw_special.append(new_number)
        else:
            draw_special[index][int(user_input[2])] = new_number
    else:
        print("\nInvalid detail type. Please enter 'd', 'n', or 's'.")
    return dates, draw_numbers, draw_special


def check_numbers_for_OCR_mistakes(numbers):
    """Sometimes the OCR will mistake a 1 for a 7, so we need to check to see if this happens.
    We can detect this since the numbers on a lottery ticket are in ascending order
    Args:
        numbers (list): A list of lists containing the extracted numbers.
    Returns:
        list: The list of lists with corrected numbers.
    """
    for draw_numbers in numbers:
        for i in range(1, len(draw_numbers)):
            last_number = draw_numbers[i-1]
            current_number = draw_numbers[i]
            if int(current_number) < int(last_number):
                if '7' in last_number[0] and '0' not in last_number[1]:
                    #then this is a 7 that is supposed to be a 1. We have to do the final check of the 0 since
                    #megamillions has numbers that go up to 70, so if we have a number that is 70+, then it is 
                    #obviously wrong
                    corrected_number = last_number.replace('7', '1')
                    draw_numbers[i-1] = corrected_number
                if '7' in last_number[1]: #then this is a 7 that is supposed to be a 1
                    corrected_number = last_number.replace('7', '1')
                    draw_numbers[i-1] = corrected_number
    
    return numbers

    
def main():
    while True:
        print("Please enter the path to your lottery ticket image (or 'exit' to quit): ")
        image_path = input().strip()
        if image_path.lower() == 'exit':
            print("Exiting the program. Goodbye!")
            break
        elif not image_path:
            print("No path entered. Please try again.")
            continue


        model = load_model("ticket_classifier_models/pt_88.52_88.89_model_weights.pth") #load a pre-trained model
        ticket_type = predict_image(model, image_path) #path to a test image
        
        dates, weekdays, draw_numbers, draw_special = read_image_text(image_path, ticket_type) #path to a test image and the predicted ticket type
    

        # print("\n[DEBUG] Before cleaning draw numbers:", draw_numbers)
        # print("[DEBUG] Before cleaning draw special:", draw_special)
        # print("[DEBUG] Before converting dates:", dates)
        draw_numbers = check_numbers_for_OCR_mistakes(draw_numbers)
        draw_numbers = clean_numbers(draw_numbers)
        draw_special = clean_numbers(draw_special)

        converted_dates = convert_date(dates)
        
        print("\nHere are the extracted details from your ticket:")
        print(f"Ticket Type: {ticket_type}")
        print(f"Draw Date (there will be 2 dates if its a range): {converted_dates}")
        print("Draw Numbers:")
        for i in range(len(draw_numbers)):
            print(f"Group {i+1}: {draw_numbers[i]}" + (f" with special numbers {draw_special[i]}"))

        while True:
            print("\nWould you like to change any of the details? (yes/no)")
            user_response = input().strip().lower()
            if user_response == 'yes':
                converted_dates, draw_numbers, draw_special = change_details(dates, draw_numbers, draw_special)
                print("\nHere are the updated details from your ticket:")
                print(f"Ticket Type: {ticket_type}")
                print(f"Draw Date (there will be 2 dates if its a range): {converted_dates}")
                print("Draw Numbers:")
                for i in range(len(draw_numbers)):
                    print(f"Group {i+1}: {draw_numbers[i]}" + (f" with special numbers {draw_special[i]}"))
            elif user_response == 'no':
                break
            else:
                print("\nInvalid response. Please enter 'yes' or 'no'.")
 
        global df
        df = abstract_funcs.convert_to_pd(f"lottery_data/{ticket_type}", df)

        for i in range(len(draw_numbers)):
            if converted_dates is None:
                numbers = draw_numbers[i]
                specials = draw_special[i]
                wins = abstract_funcs.check_for_win(ticket_type, numbers, specials, df)
            else:
                numbers = draw_numbers[i]
                specials = draw_special[i]
                try:
                    wins = abstract_funcs.check_for_win(ticket_type, numbers, specials, df, converted_dates)
                except ValueError as e:
                    print(f"\nError checking for win: {e}")

            if wins is None:
                print("\nSorry, no wins found for this group of numbers: " + str(numbers) + (f" with special numbers {specials}" if specials else ""))
            else:
                print("\nCongratulations! Here are your wins for this group of numbers: " + str(numbers) + (f" with special numbers {specials}" if specials else "") + ":")
                pprint.pprint(wins)

main()