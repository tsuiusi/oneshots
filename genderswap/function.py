import cv2
import mediapipe as mp

def resize_image(image, lcoord, rcoord):
    height, width = image.shape[:2]
    scale_factor = width // (rcoord[1] - lcoord[1])

    new_width = width // scale_factor
    new_height = height // scale_factor

    scaled_image = cv2.resize(image, (new_width, new_height))
    return scaled_image

def add_filter(image, background, lcoord, rcoord):
    foreground = resize_image(image, lcoord, rcoord) 

    x_pos, y_pos = lcoord

    # Region of interest where the image will be placed
    roi = background[y_pos:foreground.shape[0], x_pos: foreground.shape[1]]

    # normalize alpha channels from 0-255 to 0-1
    alpha_background = roi[:,:,3] / 255.0
    alpha_foreground = foreground[:,:,3] / 255.0

    # set adjusted colors
    for color in range(0, 3):
        roi[:,:,color] = alpha_foreground * foreground[:,:,color] + \
        alpha_background * roi[:,:,color] * (1 - alpha_foreground)

    # set adjusted alpha and denormalize back to 0-255
    roi[:,:,3] = (1 - (1 - alpha_foreground) * (1 - alpha_background)) * 255

    background[y_offset:y_offset+foreground.shape[0], x_offset:x_offset+foreground.shape[1]] = blended_result

    return blended_result


