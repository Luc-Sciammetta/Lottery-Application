import pandas as pd

import download_data

df = pd.DataFrame()

def convert_to_pd(file_path):
    """
    Convert a CSV file to a pandas DataFrame.
    Args:
        file_path (str): The path to the CSV file.
    Returns:
            pd.DataFrame: The converted DataFrame.
    """
    df = pd.read_csv(file_path)
    return df

def get_data_from_date(draw_date):
    """
    Get data from the DataFrame for a specific draw date.
    Args:
        draw_date (str): The draw date in 'YYYY-MM-DD'
    Returns:
        pd.Series: The row corresponding to the draw date.
        None: If no matching date is found.
    """
    for _, row in df.iterrows():
        if row['DrawDate'] == draw_date+"T08:00:00" or row['DrawDate'] == draw_date+"T07:00:00":
            return row

def check_for_matched(user_numbers, powerball, data):
    """
    Check how many numbers the user matched with the winning numbers.
    Args:
        user_numbers (list): The user's chosen numbers.
        powerball (int): The user's chosen powerball number.
        data (pd.Series): The row of data containing winning numbers.
    Returns:
        dict: A dictionary with matched numbers and powerball status.
    """
    winning_numbers = [
        data['0'],
        data['1'],
        data['2'],
        data['3'],
        data['4'],
    ]
    winning_powerball = data['Powerball']

    matched_numbers = set(user_numbers) & set(winning_numbers) #get the number of matched numbers by converting into 
    #sets and doing the intersection of them both
    has_powerball = powerball == winning_powerball #boolean whether the user has the powerball
    return {
        'matched': list(matched_numbers),
        'has_powerball': has_powerball
    }

def build_win_dict(row, matched):
    """ Build a dictionary representing the win information.
    Args:
        row (pd.Series): The row of data containing winning numbers.
        matched (dict): A dictionary with matched numbers and powerball status.
    Returns:
        dict: A dictionary with win information.
    """
    win_dict = {
        'DrawNumber': row['DrawNumber'],
        'DrawDate': row['DrawDate'],
        'MatchedNumbers': matched['matched'],
        'HasPowerball': matched['has_powerball'],
    }

    identifier = f"{len(matched['matched'])}" + (" + Powerball" if matched['has_powerball'] else "")
    if identifier == '0' or identifier == '1' or identifier == '2':
        return win_dict
    else:
        # win_dict['WinnersCount'] = row[identifier + ' Count']
        # win_dict['WinAmount'] = row[identifier + ' Amount']
        return win_dict
    
def check_for_win(user_numbers, powerball, draw_date=None):
    """
    Check if the user has won based on their numbers and an optional draw date.
    Args:
        user_numbers (list): The user's chosen numbers.
        powerball (int): The user's chosen powerball number.
        draw_date (str, optional): The draw date in 'YYYY-MM-DD'. Defaults to None.
    Returns:
        list: A list of dictionaries representing win information.
        None: If no wins are found.
    """
    if draw_date:
        row = get_data_from_date(draw_date)
        if row is None:
            return None
        
        matched_numbers = check_for_matched(user_numbers, powerball, row)
        win_dict = build_win_dict(row, matched_numbers)
        if len(win_dict['MatchedNumbers']) >= 3 or win_dict['HasPowerball'] == True:
            return [win_dict]
        else:
            return None
    else: 
        potential_wins = []
        for _, row in df.iterrows():
            matched_numbers = check_for_matched(user_numbers, powerball, row)
            win_dict = build_win_dict(row, matched_numbers)
            if len(win_dict['MatchedNumbers']) >= 3 or win_dict['HasPowerball'] == True:
                potential_wins.append(win_dict)
        
        if potential_wins == []:
            return None
        return potential_wins
    
def main():
    global df
    df = convert_to_pd("output.csv")
    numbers = [1, 2, 3, 4, 5]
    powerball = 6
    draw_date = "2024-06-01"  # Example date; can be changed or set to None
    print(check_for_win(numbers, powerball, draw_date))


main()