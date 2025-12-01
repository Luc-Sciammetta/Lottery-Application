import pandas as pd

import abstract_funcs

df = pd.DataFrame()
    
def main():
    global df
    df = abstract_funcs.convert_to_pd("lottery.csv", df)
    numbers = [6,12,28,35,66]
    powerball = 26
    draw_date = "2025-01-01"  # Example date; can be changed or set to None
    print(abstract_funcs.check_for_win(numbers, powerball, df, draw_date=draw_date))

main()