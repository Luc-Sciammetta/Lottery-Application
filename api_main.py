import pandas as pd
import pprint

from PIL import Image

import abstract_funcs
from read_text.read_image_easyocr import read_image_text
from ticket_classifiers.ticket_classifier import load_model
from ticket_classifiers.test_ticket_classifier import predict_image

df = pd.DataFrame()
HMS = False #we always want to ignore time in the comparison for this method
    
def main():
    print("Please input the path to the image you want to test:")
    # image_path = input().strip()
    image_path = "images/megamillions/IMG_3474.jpeg"

    model = load_model("ticket_classifier_models/74.18_72.22_model_weights.pth") #load a pre-trained model
    ticket_type = predict_image(model, image_path) #path to a test image

    ticket_text = read_image_text(image_path)

    print(f"Predicted Ticket Type: {ticket_type}")

    for (bbox, text, prob) in ticket_text:
        print(f'Text: {text}, Probability: {prob}')

    # global df
    # df = abstract_funcs.convert_to_pd("lottery_data/megamillions", df)
    # numbers = [19, 28, 43, 54, 61]
    # powerball = 11
    # pprint.pprint(abstract_funcs.check_for_win(numbers, powerball, df))

main()