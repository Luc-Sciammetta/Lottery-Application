import pandas as pd

import abstract_funcs
import pprint

df = pd.DataFrame()
HMS = False #we always want to ignore time in the comparison for this method
    
def main():
    global df
    df = abstract_funcs.convert_to_pd("powerball", df)
    numbers = [10, 27, 29, 31, 48]
    powerball = 22
    pprint.pprint(abstract_funcs.check_for_win(numbers, powerball, df))

main()