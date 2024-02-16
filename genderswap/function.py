import cv2
import mediapipe as mp
import numpy as np

def resize_image(image, lcoord, rcoord):
    height, width = image.shape[:2]
    scale_factor = width // (rcoord[0] - lcoord[0])

    new_width = width // scale_factor
    new_height = height // scale_factor

    scaled_image = cv2.resize(image, (new_width, new_height))
    return scaled_image

def add_filter(image, background, lcoord, rcoord):
    foreground = resize_image(image, lcoord, rcoord) 
    cv2.circle(background, lcoord, 5, (255, 0, 0))
    cv2.circle(background, rcoord, 5, (255, 0, 0))

    x_pos, y_pos = lcoord

    # Region of interest where the image will be placed, two versions for crown and for glasses
    # roi = background[y_pos - foreground.shape[0]: y_pos, x_pos: x_pos + foreground.shape[1]]
    roi = background[y_pos: y_pos + foreground.shape[0], x_pos: x_pos + foreground.shape[1]]
    height, width = roi.shape[:2]
    alpha_channel = np.ones((height, width), dtype=image.dtype) * 255
    roi = cv2.merge((roi, alpha_channel))

    # normalize alpha channels from 0-255 to 0-1
    alpha_background = roi[:,:,3] / 255.0
    alpha_foreground = foreground[:,:,3] / 255.0

    # set adjusted colors
    for color in range(0, 3):
        roi[:,:,color] = alpha_foreground * foreground[:,:,color] + \
        alpha_background * roi[:,:,color] * (1 - alpha_foreground)

    # set adjusted alpha and denormalize back to 0-255
    roi[:,:,3] = (1 - (1 - alpha_foreground) * (1 - alpha_background)) * 255
    
    # again for glasses and for crown
    # background[y_pos-foreground.shape[0]: y_pos, x_pos:x_pos+foreground.shape[1]] = roi[:, :, :3]
    background[y_pos: y_pos + foreground.shape[0], x_pos:x_pos+foreground.shape[1]] = roi[:, :, :3]


    return background



