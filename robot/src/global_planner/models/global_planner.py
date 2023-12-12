from abc import abstractmethod

import numpy as np

from global_planner.data.mask import Mask
from global_planner.data.util import normalize_image


class GlobalPlanner:
    def __init__(self, map_reader, config, convert_to_px=True):
        self.map_reader = map_reader
        self.mask = Mask(mask_type=config["mask_type"], kernel_size=config["kernel_size"], sigma=config["sigma"])
        self.probability_map = None
        self.convert_to_px = convert_to_px

    @abstractmethod
    def predict_probabilities(self, current_position, goal_position):
        pass

    def calculate_probs(self, relative_waypoints, current_position, north_heading):
        if self.convert_to_px:
            north_heading = self.map_reader.adjust_heading(current_position[0], current_position[1], north_heading)
            current_position = self.map_reader.to_px(current_position)
            relative_waypoints = self.map_reader.convert_waypoints_to_pixel_coordinates(current_position,
                                                                                            north_heading,
                                                                                            relative_waypoints)

        candidate_waypoints_crop = [self.map_reader.to_crop_coordinates(current_position, c) for c in relative_waypoints]
        probs = np.array([self.probability_map[c[1], c[0]] for c in candidate_waypoints_crop])
        return probs

    def create_masked_map(self, current_position, goal_position):
        cropped_map = self.map_reader.crop_map_by_position(current_position)
        map_tensor = normalize_image(cropped_map)
        map_tensor = np.transpose(map_tensor, (2, 0, 1)).astype(np.float32)

        current_position_crop = self.map_reader.to_crop_coordinates(current_position, current_position)
        goal_position_crop = self.map_reader.to_crop_coordinates(current_position, goal_position)
        trajectory = np.array([current_position_crop, goal_position_crop])
        masked_map = self.mask.create_masked_map(map_tensor, trajectory)
        masked_map = np.expand_dims(masked_map, axis=0)
        return masked_map


def normalize_probabilities(pred):
    min_val = pred.min()
    max_val = pred.max()
    pred_norm = (pred - min_val) / (max_val - min_val)
    return pred_norm
