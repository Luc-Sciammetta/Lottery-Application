import pandas as pd

import abstract_funcs

df = pd.DataFrame()

def main():
    global df
    df = abstract_funcs.convert_to_pd("output.csv", df)
    numbers = [1, 2, 3, 4, 5]
    powerball = 6
    draw_date = "2024-06-01"  # Example date; can be changed or set to None
    print(abstract_funcs.check_for_win(numbers, powerball, df, draw_date=draw_date, hms=True))

main()