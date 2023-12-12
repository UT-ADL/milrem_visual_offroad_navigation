import numpy as np
import cv2
from global_planner.data.util import normalize_image

def center_crop_and_resize(image_array, crop_size):
    target_width, target_height = crop_size
    # Get the dimensions of the image
    h, w, _ = image_array.shape

    # Determine the largest center crop that maintains the aspect ratio of the target size
    target_aspect = target_width / target_height
    current_aspect = w / h

    if current_aspect > target_aspect:
        # Crop the width to maintain the aspect ratio
        new_width = int(target_aspect * h)
        startx = w // 2 - new_width // 2
        crop_img = image_array[:, startx:startx + new_width]
    else:
        # Crop the height to maintain the aspect ratio
        new_height = int(w / target_aspect)
        starty = h // 2 - new_height // 2
        crop_img = image_array[starty:starty + new_height, :]

    # Resize the cropped image to the target size
    resized_img = cv2.resize(crop_img, (target_width, target_height), interpolation=cv2.INTER_AREA)

    return resized_img

def prepare_image(image_array):

    normalized_image = normalize_image(image_array)
    final_image = np.expand_dims(np.moveaxis(normalized_image, -1, 0), 0)

    return final_image
