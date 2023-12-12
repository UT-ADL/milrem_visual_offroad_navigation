
import cv2
import numpy as np
from matplotlib import cm

from viz.util import draw_direction

BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
ORANGE = (255, 102, 0)
CYAN = (0, 255, 255)


class GlobalPlannerViz:

    def __init__(self, global_planner, adjust_heading=True):
        self.map_reader = global_planner.map_reader
        # If True, current position and goal position are given in gps lan/lon coordinates and candidate trajectory in
        # relative coordinates and are converted to pixel coordinates. This typicallly used with real robot.
        # If False, all coordinates are assumed to be already in pixel coordinate and no conversion is done.
        # This is typically used with simulation environment.
        self.convert_to_px = global_planner.convert_to_px
        self.probability_map = global_planner.probability_map
        self.adjust_heading = adjust_heading

    def plot_trajectories_map(self, current_pos, goal_pos, north_heading, trajectories, trajectory_colors):
        if self.adjust_heading:
            north_heading = self.map_reader.adjust_heading(current_pos[0], current_pos[1], north_heading)

        if self.convert_to_px:
            current_pos = self.map_reader.to_px(current_pos)
            goal_pos = self.map_reader.to_px(goal_pos)

        cropped_map = self.map_reader.crop_map_by_position(current_pos)
        # Convert pixel coordinates to crop based pixel coordinates
        current_pos_crop = self.map_reader.to_crop_coordinates(current_pos, current_pos)
        goal_pos_crop = self.map_reader.to_crop_coordinates(current_pos, goal_pos)

        # Draw trajectories with different colors
        if trajectory_colors:
            for trajectory_i, trajectory in enumerate(trajectories):
                prev_wp_position = (int(current_pos_crop[0]), int(current_pos_crop[1]))
                for wp_i, wp in enumerate(trajectory):
                    if self.convert_to_px:
                        wp = self.map_reader.convert_waypoints_to_pixel_coordinates(current_pos, north_heading, [wp])
                        wp = wp.squeeze()
                    wp = self.map_reader.to_crop_coordinates(current_pos, wp)

                    traj_color = trajectory_colors[trajectory_i]
                    wp_position = (int(wp[0]), int(wp[1]))
                    cv2.line(cropped_map, prev_wp_position, wp_position, traj_color, 1)
                    cv2.circle(cropped_map, wp_position, 2, traj_color, 2)
                    prev_wp_position = wp_position

        # Draw waypoint probabilities
        prob_map_colors = self.probability_map_to_img()
        for trajectory_i, trajectory in enumerate(trajectories):
            for wp_i, wp in enumerate(trajectory):
                if self.convert_to_px:
                    wp = self.map_reader.convert_waypoints_to_pixel_coordinates(current_pos, north_heading, [wp])
                    wp = wp.squeeze()
                wp = self.map_reader.to_crop_coordinates(current_pos, wp)

                wp_color = prob_map_colors[wp[1], wp[0]]
                wp_color = (int(wp_color[0]), int(wp_color[1]), int(wp_color[2]))
                wp_position = (int(wp[0]), int(wp[1]))
                cv2.circle(cropped_map, wp_position, 1, wp_color, 2)

        # Draw start and end positions
        cv2.circle(cropped_map, (int(current_pos_crop[0]), int(current_pos_crop[1])), 2, BLUE, 2)
        cv2.circle(cropped_map, (int(goal_pos_crop[0]), int(goal_pos_crop[1])), 2, YELLOW, 2)

        draw_direction(current_pos_crop, north_heading, cropped_map, length=8, color=BLUE, thickness=2)
        return cropped_map

    def plot_probability_map(self, current_pos, goal_pos):
        if self.convert_to_px:
            current_pos = self.map_reader.to_px(current_pos)
            goal_pos = self.map_reader.to_px(goal_pos)

        prob_map_colors = self.probability_map_to_img()

        # Convert pixel coordinates to crop based pixel coordinates
        current_pos_crop = self.map_reader.to_crop_coordinates(current_pos, current_pos)
        goal_pos_crop = self.map_reader.to_crop_coordinates(current_pos, goal_pos)

        probability_map_img = prob_map_colors
        cv2.circle(probability_map_img, (int(current_pos_crop[0]), int(current_pos_crop[1])), 2, BLUE, 2)
        cv2.circle(probability_map_img, (int(goal_pos_crop[0]), int(goal_pos_crop[1])), 2, YELLOW, 2)
        return probability_map_img

    def probability_map_to_img(self):
        # Convert probabilities into colors
        prob_map_int = (255 * self.probability_map).astype(np.uint8)
        prob_map_colors = cm.Reds(prob_map_int)
        prob_map_colors = (prob_map_colors * 255).astype(np.uint8)
        prob_map_colors = cv2.cvtColor(prob_map_colors, cv2.COLOR_RGBA2RGB)
        return prob_map_colors