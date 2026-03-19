import cv2 
import numpy as np 
from matplotlib import image, pyplot as plt

import sys

def read_image(image_path):
    """ Read an image from the given path.
    Args:
        image_path (str): The path to the image file.
    Returns:
        numpy array: The image read from the file, or None if the image could not be
            read.
    """
    img = cv2.imread(image_path) #read the image using OpenCV
    if img is None:
        # print(f"[READ IMAGE]: Could not read image from {image_path}")
        sys.exit(1) #!not the best of error handling 
    return img

def get_output_dimensions(src_points):
    """
    Calculate the dimensions of the output image based on the source points.
    Args:
        src_points (numpy array): A 4x2 array of the source points in the
            original image.
    Returns:
        tuple: A tuple containing the width and height of the output image.
    """
    top_left, top_right, bottom_right, bottom_left = order_points(src_points) #order the points in a consistent way (top-left, top-right, bottom-right, bottom-left)
    width_top = np.linalg.norm(top_right - top_left) #calculate the width of the top edge
    width_bottom = np.linalg.norm(bottom_right - bottom_left) #calculate the width of the bottom edge
    output_width = int(max(width_top, width_bottom)) #the width of the output image is the maximum of the top and bottom widths (so we dont cut any image off)

    height_left = np.linalg.norm(bottom_left - top_left) #calculate the height of the left edge
    height_right = np.linalg.norm(bottom_right - top_right) #calculate the height of the right edge
    output_height = int(max(height_left, height_right)) #the height of the output image is the maximum of the left and right heights (so we dont cut any image off)

    # Lottery tickets are portrait (taller than wide)
    # If we got landscape, swap
    if output_width > output_height:
        output_width, output_height = output_height, output_width

    return output_width, output_height


def order_points(points):
    """
    Order the points in a consistent way: top-left, top-right, bottom-right, bottom-left.
    Args:
        points (numpy array): A 4x2 array of the points to be ordered.
    Returns:
        numpy array: A 4x2 array of the ordered points."""
    
    rect = np.zeros((4, 2), dtype="float32") #make a 4x2 array to hold the ordered points (these are initialized to 0)

    s = points.sum(axis=1) #calculate the sum of the x and y coordinates for each point
    rect[0] = points[np.argmin(s)] # top-left This is the smallest sum of x and y coordinates
    rect[2] = points[np.argmax(s)] # bottom-right This is the largest sum of x and y coordinates

    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)] # top-right This is the smallest difference between x and y coordinates
    rect[3] = points[np.argmax(diff)] # bottom-left This is the largest difference between x and y coordinates

    return rect


def apply_warping(image_path, src_points, dst_points, output_width, output_height):
    """
    Correct the perspective of an image given the source and destination points.
    Args:
        image_path (str): The path to the image file.
        src_points (numpy array): A 4x2 array of the source points in the
            original image.
        dst_points (numpy array): A 4x2 array of the destination points in the
            corrected image.
        output_width (int): The width of the output image.
        output_height (int): The height of the output image.
    Returns:
        numpy array: The perspective-corrected image.
    """
    
    #read the image
    image = read_image(image_path)
    
    #magical stuff happens in these lines...
    matrix = cv2.getPerspectiveTransform(src_points, dst_points) #calculates the perspective transform matrix from the source points to the destination points
    warped_image = cv2.warpPerspective(image, matrix, (output_width, output_height)) #applies the perspective transformation to the image using 
                                                                                     #the calculated matrix and the specified output dimensions

    # display the warped image
    # cv2.imshow('Warped Image', warped_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #save the warped image
    # output_path = "warped_image.jpg"
    # cv2.imwrite(output_path, warped_image)

    return warped_image


def detect_image_corners(image_path):
    """
    Detect the corners of the ticket in the image.
    Args:
        image_path (str): The path to the image file.
    Returns:
        numpy array: A 4x2 array of the detected corners of the ticket.
    """

    image = read_image(image_path) #read the image using OpenCV
    height, width = image.shape[:2] #get the height and width of the image
    min_area = (height * width) * 0.3 #set a minimum area threshold to filter out small contours

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert the image to grayscale for better contour detection

    # cv2.imshow('Blurred Image', blurred) #show the blurred image for debugging purposes
    # cv2.waitKey(0)
    
    # edged = cv2.Canny(blurred, 50, 200) #apply Canny edge detection to the blurred image to find edges in the image (this helps with contour detection)
    # cv2.imshow('Edged Image', edged) #show the edged image for debugging purposes
    # cv2.waitKey(0)

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) 
    # dilated = cv2.dilate(edged, kernel, iterations=2)
    # cv2.imshow('Dilated Image', dilated) #show the dilated image for debugging purposes
    # cv2.waitKey(0)

    # contours, hierarchy = cv2.findContours(image=dilated, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE) #finds the contours in the dilated image.

    ret, thresh = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY) #threshold the image to get a binary image. What this does is it turns all pixels with a value above 150 to white (255)
                                                                         #and all pixels with a value below 150 to black (0). This makes it easier to detect contours.
    #show the thresholded image for debugging purposes
    # cv2.imshow('Thresholded Image', thresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE) #finds the contours in the thresholded image. 
                                                                                                           #Contours are simply curves that join all the continuous points along a boundary that have the same color or intensity.
    
    contours = sorted(contours, key=cv2.contourArea, reverse=True) #sort the contours by area, largest to smallest
    # print(len(contours), "contours found") #print the number of contours found for debugging purposes
    
    ticket_corners = None
    for contour in contours[:30]: #look at the 30 largest contours (we can adjust this number if needed)
        if cv2.contourArea(contour) < min_area:
            continue #skip contours that are too small to be the ticket
        perimeter = cv2.arcLength(contour, True) #calculates the perimeter of the contour
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True) #approximate the contour to a polygon
        
        #!MAYBE CHANGE THIS BC I DONT LIKE HOW WE ASSUME ITS THE TICKET
        if len(approx) == 4: #if the approximated contour has 4 points, we can assume it's the ticket
            ticket_corners = approx.reshape(4, 2) #reshape the corners to a 4x2 array
            break

    if ticket_corners is None:
        # print("[DETECT IMAGE CORNERS]: Could not find ticket corners.")
        return None

    # Draw the 4 corners on the image so you can verify
    image_copy = image.copy()
    for point in ticket_corners:
        cv2.circle(image_copy, tuple(point), 10, (0, 255, 0), -1)
    # Draw the outline too
    cv2.drawContours(image_copy, [ticket_corners.reshape(4, 1, 2)], -1, (0, 255, 0), 3)

    # cv2.imshow('Detected Ticket', image_copy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return ticket_corners.astype(np.float32) #return the corners as a 4x2 array of floats (this is the format needed for the perspective transform)
    

def correct_image(image_path):
    """
    Correct the perspective of the image at the given path.
    Args:
        image_path (str): The path to the image file.
    Returns:
        numpy array: The perspective-corrected image, or None if the corners could not be detected or warping failed.
    """

    image = read_image(image_path)
    # print("dimensions ", image.shape)
    
    margin = 300
    src_points = detect_image_corners(image_path) #detect the corners of the ticket in the image
    corners_in_margins = 0

    if src_points is None:
        return None
    for index, point in enumerate(src_points):
        x, y = point
        if (x < margin or x > image.shape[1] - margin) and (y < margin or y > image.shape[0] - margin):
            corners_in_margins += 1
    if corners_in_margins == 4:
        #image is nicely centered, no need to warp it
        pass
    
    if src_points is not None:  
        src_points = order_points(src_points) #order the points in a consistent way (top-left, top-right, bottom-right, bottom-left)
        output_width, output_height = get_output_dimensions(src_points) #calculate the dimensions of the output image based on the source points

        #define the destination points for the perspective transform (these are the corners of the output image)
        dst_points = np.array([ 
            [0, 0],                          # top-left
            [output_width - 1, 0],           # top-right
            [output_width - 1, output_height - 1],  # bottom-right
            [0, output_height - 1]           # bottom-left
        ], dtype=np.float32)

        corrected_image = apply_warping(image_path, src_points, dst_points, output_width, output_height) #correct the perspective of the image using the detected corners and the defined destination points

        if corrected_image is None:
            # print("[CORRECT IMAGE]: Could not apply warping to the image")
            return None

        return corrected_image
    
    return None


if __name__ == "__main__":
    image_path = "images/megamillions/IMG_3468.jpeg"

    correct_image(image_path)
