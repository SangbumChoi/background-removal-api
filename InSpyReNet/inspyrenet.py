from transparent_background import Remover
from PIL import Image
import numpy as np
from skimage.measure import find_contours
from shapely.geometry import Point, Polygon
from skimage.draw import polygon as skpolygon
import matplotlib.pyplot as plt
import cv2


# Initialize the model globally
remover = Remover(jit=False)

def process_image(input_image, output_type):
    global remover
    
    if output_type == "Mask only":
        # Process the image and get only the mask
        output = remover.process(input_image, type='map')
        if isinstance(output, Image.Image):
            # If output is already a PIL Image, convert to grayscale
            mask = output.convert('L')
        else:
            # If output is a numpy array, convert to PIL Image
            mask = Image.fromarray((output * 255).astype(np.uint8), mode='L')
        return mask
    else:
        # Process the image and return the RGBA result
        output = remover.process(input_image, type='rgba')
        return output
    
def check_alpha_layer(image, x, y):
    # Extract the alpha channel
    alpha = image.split()[-1]
    # Get the alpha value at the clicked point
    alpha_value = alpha.getpixel((x, y))
    # Check if the point is within the mask (alpha value is not fully transparent)
    is_in_mask = alpha_value > 0
    return is_in_mask, alpha


def get_polygon_from_mask(image, mask, x, y):
    # Convert the image to a NumPy array
    image_array = np.array(image)
    zero_mask = np.zeros(image_array.shape[:2], dtype=np.uint8)
    mask = np.uint8(np.array(mask) > 0)

    threshold_area = 1000     #threshold area 
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    for contour in contours:        
        area = cv2.contourArea(contour)         
        if area > threshold_area:
            result = cv2.pointPolygonTest(contour, (x,y), False) 
            if result == 1.0:
                print('test')
                break

    x,y,w,h = cv2.boundingRect(contour)
    bounding_box = [x,y,x+w,y+h]

    return contour, bounding_box


if __name__ == "__main__":
    # Read the image file 
    image = Image.open('./examples/example1.jpg')
    output = process_image(input_image=image, output_type='default')
    # Extract the alpha channel
    alpha = output.split()[-1]
    print(alpha)
    print(alpha.getpixel((1280, 900)))
    print(check_alpha_layer(alpha, 1280, 900))
    print(get_polygon_from_mask(image=image, mask=alpha, x=1280, y=900))