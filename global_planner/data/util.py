import numpy as np


def normalize_image(image):
    # Convert image to float32 to avoid data type issues during computation
    image = image.astype(np.float32)
    image = image / 255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized_image = (image - mean) / std
    return normalized_image
