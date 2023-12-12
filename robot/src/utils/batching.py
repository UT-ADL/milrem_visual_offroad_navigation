# batch observation images + context images
import numpy as np
from utils.preprocessing_images import center_crop_and_resize, prepare_image

def batch_obs_plus_context(buffer_length, waypoint_spacing, deque_images, fps, crop_size):
    
    # images arriving in 15 Hz freq
    if fps == 15:
        batch_images = []
        for i in range(0, buffer_length, waypoint_spacing):            
            resized_image = center_crop_and_resize(deque_images[i], crop_size)
            preprocessed_image = prepare_image(resized_image)
            # batch_images.append(deque_images[i])
            batch_images.append(preprocessed_image)
        
        obs_plus_context_imgs = np.concatenate(batch_images, axis=1)
    
    # images arriving in 4 Hz freq
    else:
        obs_plus_context_imgs = np.concatenate(deque_images, axis=1)
    
    
    return obs_plus_context_imgs