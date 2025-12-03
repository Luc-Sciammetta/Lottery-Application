import pandas as pd

import abstract_funcs

df = pd.DataFrame()
HMS = False #we always want to ignore time in the comparison for this method
    
def main():
    global df
    df = abstract_funcs.convert_to_pd("powerball", df)
    numbers = [5,18,26,47,59]
    powerball = 1
    draw_date = "2025-12-01"  # Example date; can be changed or set to None
    print(abstract_funcs.check_for_win(numbers, powerball, df, draw_date=draw_date))
    print()
    print(abstract_funcs.check_for_win(numbers, powerball, df))


main()