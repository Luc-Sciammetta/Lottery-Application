import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

def read_image(image_path):
    try:
        text = pytesseract.image_to_string('image_path', timeout=2)
        print(text)
    except RuntimeError as timeout_error:
        print(f"Timeout Reading image {image_path}")
        return ""

    return text

# # Get bounding box estimates
# print(pytesseract.image_to_boxes(Image.open('test.png')))

# # Get verbose data including boxes, confidences, line and page numbers
# print(pytesseract.image_to_data(Image.open('test.png')))

# # Get information about orientation and script detection
# print(pytesseract.image_to_osd(Image.open('test.png')))


# # getting multiple types of output with one call to save compute time
# # currently supports mix and match of the following: txt, pdf, hocr, box, tsv
# text, boxes = pytesseract.run_and_get_multiple_output('test.png', extensions=['txt', 'box'])