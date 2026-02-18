import pandas as pd

import abstract_funcs
from read_text.read_image_easyocr import read_image_text

#if using a non-pre-trained model, use these imports
# from ticket_classifiers.ticket_classifier import load_model
# from ticket_classifiers.test_ticket_classifier import predict_image

#if using a pre-trained model, use these imports
from ticket_classifiers.pretrained_ticket_classifier import load_model, predict_image

df = pd.DataFrame()
HMS = False #we always want to ignore time in the comparison for this method
    
def main():
    image_path = "images/euromillions/AI_img1.png"

    model = load_model("ticket_classifier_models/pt_88.52_88.89_model_weights.pth") #load a pre-trained model
    ticket_type = predict_image(model, image_path) #path to a test image

    ticket_text = read_image_text(image_path)

    # print(f"Predicted Ticket Type: {ticket_type}")

    # for (bbox, text, prob) in ticket_text:
    #     print(f'Text: {text}, Probability: {prob}')

    # global df
    # df = abstract_funcs.convert_to_pd("lottery_data/megamillions", df)
    # numbers = [19, 28, 43, 54, 61]
    # powerball = 11
    # pprint.pprint(abstract_funcs.check_for_win(numbers, powerball, df))

main()