from abc import abstractmethod
from collections import deque

import cv2
import numpy as np
from tqdm import tqdm

from viz.util_camera import load_camera_model, to_camera_frame

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
FONT_SIZE = 0.3

FRAME_SIZE = (1280, 720)


class BaseVisualizer:
    TOP_DOWN_VIEW_SCALE = 15
    GOAL_SPACING = 75

    def __init__(self, goal_conditioning=False):
        self.camera_model = load_camera_model()
        self.goal_conditioning = goal_conditioning
        self.true_distance = 0.0
        self.goal_idx = 0
        self.sample_new_goal = True
        self.sampled_goals = 0
        self.completed_goals = 0
        self.num_waypoints_displayed = 5

    @abstractmethod
    def predict(self, model, obs_tensor, goal_tensor):
        """
        Override this model specific method by making prediction using given model, observation and goal.

        :param model: Model to use for predictions
        :param obs_tensor: Observation tensor
        :param goal_tensor: Goal tensor
        :return: predicted actions (num_trajectories, num_waypoints, 2), predicted distance (1)
        """
        pass

    def draw_extras(self, img):
        """
        Override this method if addition model specific drawing has to be done onto the frame,

        :param img: Image of current frame to draw onto
        """
        pass

    def create_frame(self, model, dataset, frame_index):
        obs_tensor, _, labels, data, obs_img, _ = dataset[frame_index]
        rectified_image = self.rectify_image(obs_img)

        if self.goal_conditioning and self.sample_new_goal:
            last_obs_tensor, _, _, last_data, goal_img, _ = dataset[frame_index + self.GOAL_SPACING]

            self.goal_tensor = last_obs_tensor[-3:]
            self.goal_img = goal_img
            self.goal_idx = last_data["idx"]
            self.sample_new_goal = False

        if not self.goal_conditioning:
            last_obs_tensor, _, _, last_data, goal_img, _ = dataset[len(dataset) - 1]
            self.goal_tensor = last_obs_tensor[-3:]
            self.goal_img = goal_img
            self.goal_idx = last_data["idx"]

        self.true_distance = (self.goal_idx - data['idx']) / 3.75  # TODO: remove hardcoded value

        predicted_actions, predicted_dist = self.predict(model, obs_tensor, self.goal_tensor)

        if predicted_dist < 2.0:
            self.sample_new_goal = True
            self.sampled_goals += 1
            self.completed_goals += 1

        if frame_index > self.goal_idx:
            self.sampled_goals += 1
            self.sample_new_goal = True

        # Shorten trajectories to 5 so it would be consistent with all models (NoMaD has trajectory of 5)
        # Use only x, y coordinates and discard angles
        predicted_actions = predicted_actions[:, :self.num_waypoints_displayed, :2]
        label_actions = labels[0][:self.num_waypoints_displayed, :]

        self.draw_extras(rectified_image)

        to_camera_frame(rectified_image, label_actions, GREEN)
        for i in range(len(predicted_actions)):
            to_camera_frame(rectified_image, predicted_actions[i], RED)
        self.draw_info_overlay(rectified_image, data, predicted_dist)
        self.draw_top_town_overlay(rectified_image, None, predicted_actions.squeeze(), label_actions)
        self.draw_waypoint_img(rectified_image, self.goal_img)

        return rectified_image

    def draw_info_overlay(self, frame, data, pred_distance):

        self.draw_rectangle(frame, 5, 15, 90, 130)

        cv2.putText(frame, 'obs idx: {}'.format(data['idx']), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, GREEN, 1, cv2.LINE_AA)
        cv2.putText(frame, 'goal idx: {}'.format(self.goal_idx), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, GREEN, 1, cv2.LINE_AA)

        cv2.putText(frame, 'true dist: {:.1f}'.format(self.true_distance), (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, GREEN, 1, cv2.LINE_AA)

        if pred_distance:
            cv2.putText(frame, 'pred dist: {:.1f}'.format(pred_distance), (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, RED, 1, cv2.LINE_AA)

        cv2.putText(frame, 'goals: {}/{}'.format(self.completed_goals, self.sampled_goals), (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, GREEN, 1, cv2.LINE_AA)

    def draw_top_town_overlay(self, frame, waypoint_samples, predicted_wps, true_wps):
        self.draw_rectangle(frame, 10, 210, 140, 130)

        current_position = (75, 325)

        if waypoint_samples is not None:
            for i in range(waypoint_samples.shape[0]):
                wp_trajectory = waypoint_samples[i]
                self.draw_top_down_trajectory(frame, wp_trajectory, current_position, BLUE)

        predicted_wps = predicted_wps.reshape(-1, 2)
        self.draw_top_down_trajectory(frame, predicted_wps, current_position, RED)
        self.draw_top_down_trajectory(frame, true_wps, current_position, GREEN)
        cv2.circle(frame, current_position, 2, (0, 255, 255), 2)

    def draw_top_down_trajectory(self, frame, predicted_wps, current_position, color):
        for i in range(predicted_wps.shape[0]):
            pred_wp = predicted_wps[i]
            scaled_current_wp = (current_position[0] - int(self.TOP_DOWN_VIEW_SCALE * pred_wp[1]),
                                 current_position[1] - int(self.TOP_DOWN_VIEW_SCALE * pred_wp[0]))
            cv2.circle(frame, scaled_current_wp, 2, color, 2)

            if i > 0:
                pred_prev_wp = predicted_wps[i - 1]
            else:
                pred_prev_wp = [0.0, 0.0]

            scaled_prev_wp = (current_position[0] - int(self.TOP_DOWN_VIEW_SCALE * pred_prev_wp[1]),
                              current_position[1] - int(self.TOP_DOWN_VIEW_SCALE * pred_prev_wp[0]))
            cv2.line(frame, scaled_prev_wp, scaled_current_wp, color, 2)

    def draw_waypoint_img(self, frame, waypoint_img):
        waypoint_img = cv2.cvtColor(np.array(waypoint_img), cv2.COLOR_RGB2BGR)

        scale_percent = 20  # percent of original size
        width = int(waypoint_img.shape[1] * scale_percent / 100)
        height = int(waypoint_img.shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image
        resized = cv2.resize(waypoint_img, dim, interpolation=cv2.INTER_AREA)

        spacing = 10
        frame[spacing:resized.shape[0]+spacing, frame.shape[1]-resized.shape[1]-spacing:frame.shape[1]-spacing] = resized

    def draw_rectangle(self, frame, x, y, w, h):
        sub_img = frame[y:y + h, x:x + w]
        rect = np.full(sub_img.shape, (0, 0, 0), np.uint8)
        alpha = 0.4
        res = cv2.addWeighted(sub_img, alpha, rect, 1 - alpha, 0)
        frame[y:y + h, x:x + w] = res

    def show_next_frame(self, model, dataset, deq):
        frame_index = deq[0]
        image = self.create_frame(model, dataset, frame_index)
        cv2.imshow('vis', image)

    def create_video(self, dataset, model, output_file_name, start_frame=0, codec='avc1'):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_file_name, fourcc, 15, FRAME_SIZE)

        for frame_index in tqdm(range(start_frame, len(dataset) - self.GOAL_SPACING)):
            img = self.create_frame(model, dataset, frame_index)
            img = cv2.resize(img, FRAME_SIZE, interpolation=cv2.INTER_LINEAR)
            out.write(img)

        out.release()

    def create_interactive(self, dataset, model, start_frame):
        deq = deque(range(0, len(dataset) - self.GOAL_SPACING))
        deq.rotate(-start_frame)
        cv2.namedWindow('vis', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('vis', FRAME_SIZE[0], FRAME_SIZE[1])
        self.show_next_frame(model, dataset, deq)
        while cv2.getWindowProperty('vis', cv2.WND_PROP_VISIBLE) >= 1:
            pressed_key = cv2.waitKey(10)
            if pressed_key == ord('j'):
                deq.rotate(1)
                self.show_next_frame(model, dataset, deq)
            elif pressed_key == ord('k'):
                deq.rotate(-1)
                self.show_next_frame(model, dataset, deq)
            elif pressed_key == ord('u'):
                deq.rotate(30)
                self.show_next_frame(model, dataset, deq)
            elif pressed_key == ord('i'):
                deq.rotate(-30)
                self.show_next_frame(model, dataset, deq)

    def rectify_image(self, obs_img):
        obs_img = cv2.cvtColor(np.array(obs_img), cv2.COLOR_RGB2BGR)
        rectified_image = np.zeros((obs_img.shape[0], obs_img.shape[1], 3), dtype=np.uint8)
        self.camera_model.rectifyImage(raw=obs_img, rectified=rectified_image)
        return rectified_image
