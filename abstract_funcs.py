import pandas as pd

def get_data_from_date(draw_date, df, hms=False):
    """
    Get data from the DataFrame for a specific draw date.
    Args:
        draw_date (str): The draw date in 'YYYY-MM-DD'
        df (pd.DataFrame): The DataFrame containing lottery data.
        hms (bool): Whether to consider time in the comparison. Defaults to False.
    Returns:
        pd.Series: The row corresponding to the draw date.
        None: If no matching date is found.
    """
    for _, row in df.iterrows():
        if hms:
            if row['drawing_date'] == draw_date+"T08:00:00" or row['drawing_date'] == draw_date+"T07:00:00":
                return row
        else: 
            if row['drawing_date'] == draw_date:
                return row

def convert_to_pd(game, df):
    """
    Convert a CSV file to a pandas DataFrame.
    Args:
        game (str): The name of the game corresponding to the CSV file.
        df (pd.DataFrame): The DataFrame to populate.
    Returns:
            pd.DataFrame: The converted DataFrame.
    """
    df = pd.read_csv(f"{game}.csv")
    return df

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
        data['ball1'],
        data['ball2'],
        data['ball3'],
        data['ball4'],
        data['ball5'],
    ]
    winning_powerball = data['powerball']

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
        # 'DrawNumber': row['DrawNumber'],
        'drawing_date': row['drawing_date'],
        'matched_numbers': matched['matched'],
        'number_of_matched': len(matched['matched']),
        'has_powerball': matched['has_powerball'],
    }

    return win_dict

    # identifier = f"{len(matched['matched'])}" + (" + Powerball" if matched['has_powerball'] else "")
    # if identifier == '0' or identifier == '1' or identifier == '2':
    #     return win_dict
    # else:
        # win_dict['WinnersCount'] = row[identifier + ' Count']
        # win_dict['WinAmount'] = row[identifier + ' Amount']
    #     return win_dict

def check_for_win(user_numbers, powerball, df, draw_date=None, hms=False):
    """
    Check if the user has won based on their numbers and an optional draw date.
    Args:
        user_numbers (list): The user's chosen numbers.
        powerball (int): The user's chosen powerball number.
        df (pd.DataFrame): The DataFrame containing lottery data.
        draw_date (str, optional): The draw date in 'YYYY-MM-DD'. Defaults to None.
        hms (bool): Whether to consider time in the comparison. Defaults to False.
    Returns:
        list: A list of dictionaries representing win information.
        None: If no wins are found.
    """
    if draw_date:
        row = get_data_from_date(draw_date, df, hms)
        if row is None:
            return None
        
        matched_numbers = check_for_matched(user_numbers, powerball, row)
        win_dict = build_win_dict(row, matched_numbers)
        if len(win_dict['matched_numbers']) >= 3 or win_dict['has_powerball'] == True:
            return [win_dict]
        else:
            return None
    else: 
        potential_wins = []
        for _, row in df.iterrows():
            matched_numbers = check_for_matched(user_numbers, powerball, row)
            win_dict = build_win_dict(row, matched_numbers)
            if len(win_dict['matched_numbers']) >= 3 or win_dict['has_powerball'] == True:
                potential_wins.append(win_dict)
        
        if potential_wins == []:
            return None
        return potential_wins