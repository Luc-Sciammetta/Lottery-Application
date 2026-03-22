import pandas as pd
import datetime
from get_lottery_data_files.get_dataset_from_api import make_dataset

lowest_to_win = {
    'powerball': [[5, 1], [5, 0], [4, 1], [4, 0], [3, 1], [3, 0], [2, 1], [1, 1], [0, 1]],
    'megamillions': [[5, 1], [5, 0], [4, 1], [4, 0], [3, 1], [3, 0], [2, 1], [1, 1], [0, 1]],
    'lottoamerica': [[5, 1], [5, 0], [4, 1], [4, 0], [3, 1], [3, 0], [2, 1], [1, 1], [0, 1]],
    'euromillions': [[5, 2], [5, 1], [5, 0], [4, 2], [4, 1], [3, 2], [4, 0], [2, 2], [3, 1], [3, 0], [1, 2], [2, 1], [2, 0]]
}

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

    df = pd.read_csv(f"lottery_data/{game}.csv")

    if df.iloc[-1, 0] != datetime.datetime.now().strftime("%Y-%m-%d"): #update the csv file and remake the dataframe
        make_dataset(game)
        df = pd.read_csv(f"lottery_data/{game}.csv")

    return df

def check_for_matched(game, user_numbers, specials, data):
    """
    Check how many numbers the user matched with the winning numbers.
    Args:
        game (str): The name of the game.
        user_numbers (list): The user's chosen numbers.
        specials (list): The user's chosen special numbers.
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
    
    # print(data)
    if game == 'euromillions':
        winning_specials = [data['special1'], data['special2']]
    else:
        winning_specials = [data['special']]

    matched_numbers = set(user_numbers) & set(winning_numbers) #get the number of matched numbers by converting into 
    #sets and doing the intersection of them both
    matched_specials = set(specials) & set(winning_specials)
    return {
        'matched_numbers': list(matched_numbers),
        'matched_specials': list(matched_specials)
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
        'matched_numbers': matched['matched_numbers'],
        'number_of_matched_numbers': len(matched['matched_numbers']),
        'matched_specials': matched['matched_specials'],
        'number_of_matched_specials': len(matched['matched_specials'])
    }

    return win_dict

    # identifier = f"{len(matched['matched'])}" + (" + Powerball" if matched['has_powerball'] else "")
    # if identifier == '0' or identifier == '1' or identifier == '2':
    #     return win_dict
    # else:
        # win_dict['WinnersCount'] = row[identifier + ' Count']
        # win_dict['WinAmount'] = row[identifier + ' Amount']
    #     return win_dict

def check_for_win(game, user_numbers, specials, df, draw_date_s=None, hms=False):
    """
    Check if the user has won based on their numbers and an optional draw date.
    Args:
        game (str): The name of the game corresponding to the CSV file.
        user_numbers (list): The user's chosen numbers. NOTE: This can be None
        specials (list): The user's chosen special numbers. NOTE: This can be None
        df (pd.DataFrame): The DataFrame containing lottery data.
        draw_date_s (list, optional): A list of draw dates in 'YYYY-MM-DD'. Defaults to None.
        hms (bool): Whether to consider time in the comparison. Defaults to False.
    Returns:
        list: A list of dictionaries representing win information.
        None: If no wins are found.
    """
    matches_needed = lowest_to_win[game]

    if draw_date_s is None:
        draw_date_s = []
    
    if len(draw_date_s) == 2: #ticket is valid for an interval of time
        potential_wins = []
        for _, row in df.iterrows():
            if row['drawing_date'] >= draw_date_s[0] and row['drawing_date'] <= draw_date_s[1]:
                matched = check_for_matched(game, user_numbers, specials, row)
                win_dict = build_win_dict(row, matched)
                # print(win_dict)
                if [win_dict['number_of_matched_numbers'], win_dict['number_of_matched_specials']] in matches_needed:
                    potential_wins.append(win_dict)
        if potential_wins == []:
            return None
        return potential_wins

    elif len(draw_date_s) == 1: #ticket is valid for a specific draw date
        row = get_data_from_date(draw_date_s[0], df, hms)
        if row is None:
            return None
        
        matched = check_for_matched(game, user_numbers, specials, row)
        win_dict = build_win_dict(row, matched)

        if [win_dict['number_of_matched_numbers'], win_dict['number_of_matched_specials']] in matches_needed:
            return [win_dict]
        
    elif len(draw_date_s) == 0: #no draw date provided
        potential_wins = []
        for _, row in df.iterrows():
            matched = check_for_matched(game, user_numbers, specials, row)
            win_dict = build_win_dict(row, matched)
            if [win_dict['number_of_matched_numbers'], win_dict['number_of_matched_specials']] in matches_needed:
                potential_wins.append(win_dict) 
        if potential_wins == []:
            return None
        return potential_wins
    
    else:
        raise ValueError("Invalid number of draw dates provided. Must be 0, 1, or 2.")