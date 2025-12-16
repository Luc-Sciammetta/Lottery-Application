import requests
import pandas as pd

#these are the possible games that can be used:
# "powerball" --> Powerball
# "megamillions" --> Mega Millions
# "lucky" --> Lucky4Life
# "cashpop" --> Cash Pop
# "cash4life" --> Cash4Life
# "lottoamerica" --> Lotto America
# "euromillions" --> EuroMillions
# "eurojackpot" --> EuroJackpot

url = "https://api.lotterydata.io/"
paths = {
        "latest": "{game}/v1/latest",
        "by_date": "/{game}/v1/drawing/{date}",
        "between_dates": "/{game}/v1/betweendates/{first_date}/{second_date}",
        "latest_90": "/{game}/v1/latest90",
        "latest_10": "/{game}/v1/latest10",
        "check_ticket": "/{game}/v1/checkticket/{drawing_date}/{num1}/{num2}/{num3}/{num4}/{num5}/{pb}"
    }

headers = {
    "x-api-key": "qlGVGlc0BIaCX9uysqtA1a7KXILyaiIN4K6koJoS"
}

def get_response(url, headers):
    """ Get JSON response from the API. 
    Args:
        url (str): The API endpoint URL.
        headers (dict): The headers for the request.
    Returns:
        dict: The JSON response from the API."""
    response = requests.get(url, headers=headers)
    return response.json()

def make_dataset(game, start_date="2025-01-01", end_date="2027-01-01"):
    """ Create the dataset and save it to a CSV file. 
    Args:
        game (str): The lottery game to fetch data for.
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.
    Returns:
        pd.DataFrame: The created dataset as a DataFrame."""
    json_data = get_response(url+paths["between_dates"].format(game=game, first_date=start_date, second_date=end_date), headers)
    df = pd.DataFrame(json_data['data'])
    df = df.drop(columns=['video_url', 'number_set', "next_jackpot", "next_drawing_date"], axis = 1)

    df['jackpot'] = (df['jackpot']
                    .str.replace("$", "", regex=False)
                    .str.replace(",", "", regex=False)
                    .str.replace(".", "", regex=False)
                    .astype(int))
    df['estimated_cash_value'] = (df['estimated_cash_value']
                    .str.replace("$", "", regex=False)
                    .str.replace(",", "", regex=False)
                    .str.replace(".", "", regex=False)
                    .astype(int))
    
    df = df.iloc[::-1].reset_index(drop=True)

    # df.rename(columns={'ball1': 1, 'ball2': 2, 'ball3': 3, 'ball4': 4, 'ball5': 5}, inplace=True)

    df.to_csv(f"{game}.csv", index=False)
    print(df)

    return df

def add_dates_to_dataset(game, start_date, end_date):
    """ Add data for specific date range to the existing dataset.
    Args:
        game (str): The lottery game to fetch data for.
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.
    Returns:
        pd.DataFrame: The updated dataset as a DataFrame.
        """
    json_data = get_response(url+paths["between_dates"].format(game=game, first_date=start_date, second_date=end_date), headers)
    df_new = pd.DataFrame(json_data['data'])
    df_new = df_new.drop(columns=['video_url', 'number_set', "next_jackpot", "next_drawing_date"], axis = 1)

    df_new['jackpot'] = (df_new['jackpot']
                    .str.replace("$", "", regex=False)
                    .str.replace(",", "", regex=False)
                    .str.replace(".", "", regex=False)
                    .astype(int))
    df_new['estimated_cash_value'] = (df_new['estimated_cash_value']
                    .str.replace("$", "", regex=False)
                    .str.replace(",", "", regex=False)
                    .str.replace(".", "", regex=False)
                    .astype(int))
        
    df_existing = pd.read_csv(f"{game}.csv")
    df_combined = pd.concat([df_existing, df_new]).drop_duplicates().reset_index(drop=True)

    df_combined['drawing_date'] = pd.to_datetime(df_combined['drawing_date'])
    df_combined = df_combined.sort_values('drawing_date').reset_index(drop=True)

    df_combined.to_csv(f"{game}.csv", index=False)
    print(df_combined)
    return df_combined

def get_dataset(game):
    """ Get the dataset from the CSV file.
    Args:
        game (str): The lottery game to fetch data for.
    Returns:
        pd.DataFrame: The dataset as a DataFrame.
    """
    df = pd.read_csv(f"{game}.csv")
    df['drawing_date'] = pd.to_datetime(df['drawing_date'])
    return df


def main():
    make_dataset("euromillions")
    # add_dates_to_dataset("powerball", "2024-01-01", "2025-01-01")
    # print(get_dataset("powerball"))
    pass
main()