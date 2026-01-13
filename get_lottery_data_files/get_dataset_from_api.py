import requests
import pandas as pd
from pandas.api.types import is_string_dtype

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
    "x-api-key": "IImNy8qC9d9YxEY22bcsW9jUvhPCYQqh9dSsZI8C"
}

features = {
    "powerball": ["drawing_date","ball1","ball2","ball3","ball4","ball5","powerball","multiplier","jackpot","estimated_cash_value"],
    "megamillions": ["drawing_date","ball1","ball2","ball3","ball4","ball5","megaball","jackpot","estimated_cash_value","multiplier"],
    "euromillions": ["drawing_date","ball1","ball2","ball3","ball4","ball5","star1","star2","jackpot","prizes"],
    "eurojackpot": ["drawing_date","ball1","ball2","ball3","ball4","ball5","euro1","euro2"], #jackpot_millions? marketing_jackpot_millions? special_marketing_jackpot_millions?
    "lottoamerica": ["drawing_date","ball1","ball2","ball3","ball4","ball5","starball","bonus","jackpot"]
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

    if 'message' in json_data:
        print(f"Error fetching data: {json_data['message']}")
        return pd.DataFrame()  # Return empty DataFrame on error

    df = pd.DataFrame(json_data['data'])

    df = df.loc[:, df.columns.intersection(features[game])]
    df['drawing_date'] = pd.to_datetime(df['drawing_date'])

    
    for col in df.columns:
        # skip known special columns
        if col in {"prizes", "drawing_date"}:
            continue

        # only apply string operations to string columns
        if is_string_dtype(df[col]):
            df[col] = (
                df[col]
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.replace(".", "", regex=False)
                .astype(int)
            )

    df = df.iloc[::-1].reset_index(drop=True)

    df.to_csv(f"lottery_data/{game}.csv", index=False)
    print(df)

    return df

def get_dataset(game):
    """ Get the dataset from the CSV file.
    Args:
        game (str): The lottery game to fetch data for.
    Returns:
        pd.DataFrame: The dataset as a DataFrame.
    """
    df = pd.read_csv(f"lottery_data/{game}.csv")
    df['drawing_date'] = pd.to_datetime(df['drawing_date'])
    return df

def main():
    make_dataset("powerball")
    make_dataset("megamillions")
    make_dataset("euromillions")
    make_dataset("lottoamerica")
    # print(get_dataset("powerball"))

main()