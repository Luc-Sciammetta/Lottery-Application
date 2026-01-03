import pandas as pd

import abstract_funcs
import pprint

df = pd.DataFrame()
HMS = False #we always want to ignore time in the comparison for this method
    
def main():
    global df
    df = abstract_funcs.convert_to_pd("lottery_data/powerball", df)
    numbers = [19, 28, 43, 54, 61]
    powerball = 11
    pprint.pprint(abstract_funcs.check_for_win(numbers, powerball, df))

main()