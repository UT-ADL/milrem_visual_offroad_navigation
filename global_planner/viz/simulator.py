import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from data.dataset import tracks_mapping, OneHotEncodedDataset
from data.mapping import MapReader
from data.mask import Mask
from models.global_planner_onnx import GlobalPlannerOnnx
from viz.global_planner_viz import GlobalPlannerViz, GREEN
from viz.util import calculate_distance, calculate_angle

GOAL_THRESHOLD = 15.0
NUM_SAMPLED_WAYPOINTS = 10


def parse_arguments():
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--dataset-name",
        "-d",
        type=str,
        required=True,
        help="Path to model checkpoint.",
    )

    argparser.add_argument(
        "--map-type",
        "-t",
        type=str,
        default="baseelev",
        choices=["orienteering", "baseelev"],
        help="Path to model checkpoint.",
    )

    argparser.add_argument(
        "--model-path",
        "-m",
        type=str,
        required=True,
        help="Path to model checkpoint.",
    )

    argparser.add_argument(
        "--config-path",
        "-c",
        type=str,
        required=True,
        help="Path to configuration file.",
    )

    argparser.add_argument(
        "--output-file",
        "-o",
        type=str,
        required=False,
        help="Path to model checkpoint.",
    )

    argparser.add_argument(
        "--max-frames",
        "-f",
        type=int,
        default=150,
        required=False,
        help="Max number of frames in video."
    )

    return argparser.parse_args()


def get_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    with open("config/env.yaml", "r") as f:
        env_config = yaml.safe_load(f)
        config.update(env_config)

    return config


def create_mask(config):
    return Mask(
        mask_type=config["mask_type"],
        kernel_size=config["kernel_size"],
        sigma=config["sigma"])


def create_dataset(dataset_name, map_reader, config):
    dataset = OneHotEncodedDataset(Path(config["dataset_path"]) / dataset_name,
                                   map_reader,
                                   map_size=config["map_size"],
                                   trajectory_min_len=config['trajectory_sim_length'],
                                   trajectory_max_len=config['trajectory_sim_length'],
                                   trajectory_sampling_rate=config['trajectory_sampling_rate'],
                                   config=config)

    return dataset


class MapSimulator:
    def __init__(self, segment_dataset, map_reader, model_path, config):
        self.global_planner = GlobalPlannerOnnx(map_reader, config, model_path, convert_to_px=False)
        self.map_reader = map_reader
        self.segment_dataset = segment_dataset
        self.map_size = segment_dataset.map_size
        self.start_index = 0

        _, _, trajectory, _ = self.segment_dataset[self.start_index]

        self.current_position = self.segment_dataset.positions[self.start_index].detach().numpy()
        self.position_history = []
        self.position_history.append(self.current_position)

        self.goal_index = self.start_index + len(trajectory) - 1
        self.goal_position = self.segment_dataset.positions[self.goal_index].detach().numpy()
        self.distance_to_goal = calculate_distance(self.current_position, self.goal_position)
        self.calculate_direction_angle()

    def step(self, step_size):
        self.start_index += step_size
        if self.calculate_next_goal():
            return

        # Sample random waypoints and calculate probability score for each
        candidate_waypoints = sample_waypoints(self.current_position, self.angle, NUM_SAMPLED_WAYPOINTS)
        self.global_planner.predict_probabilities(self.current_position, self.goal_position)
        waypoint_probs = self.global_planner.calculate_probs(candidate_waypoints,
                                                             self.current_position,
                                                             self.goal_position)
        result_img = self.draw_current_state(candidate_waypoints)

        # Change current position to waypoint with best score
        self.current_position = candidate_waypoints[waypoint_probs.argmax()]
        self.position_history.append(self.current_position)

        return result_img

    def calculate_next_goal(self):
        self.distance_to_goal = calculate_distance(self.current_position, self.goal_position)
        if self.distance_to_goal < GOAL_THRESHOLD:
            # Goal is reached, getting next goal
            self.start_index = self.goal_index
            _, _, trajectory, _ = self.segment_dataset[self.start_index]
            self.goal_index = self.goal_index + len(trajectory) - 1

            if self.goal_index > len(self.segment_dataset):
                # Final goal reached
                return True

            self.goal_position = self.segment_dataset.positions[self.goal_index].detach().numpy()
        self.calculate_direction_angle()
        return False

    def draw_current_state(self, candidate_waypoints):
        global_planner_viz = GlobalPlannerViz(self.global_planner, adjust_heading=False)
        trajectories_map_img = global_planner_viz.plot_trajectories_map(self.current_position, self.goal_position,
                                                                        self.north_heading(), [candidate_waypoints], None)
        probability_map_img = global_planner_viz.plot_probability_map(self.current_position, self.goal_position)
        result_img = np.concatenate((trajectories_map_img, probability_map_img), axis=1)
        result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        self.draw_trajectory_history(trajectories_map_img)
        return result_img

    def north_heading(self):
        return self.angle + 90

    def calculate_direction_angle(self):
        if len(self.position_history) == 1:
            # Robot angle is initialized to the direction of goal
            self.angle = calculate_angle(self.current_position, self.goal_position)
        else:
            if self.distance_to_goal > self.map_size:
                # Going away from the goal, return back to the direction of goal
                self.angle = calculate_angle(self.current_position, self.goal_position)
            else:
                # Direction is preserved as angle between previous two positions
                self.angle = calculate_angle(self.position_history[-2], self.current_position)

    def draw_trajectory_history(self, trajectories_map_img):
        for pos in self.position_history[-11:-1]:
            pos_map = pos - self.current_position + self.map_size
            cv2.circle(trajectories_map_img, (int(pos_map[0]), int(pos_map[1])), 1, GREEN, 2)

    def create_video(self, output_file, max_frames, config, codec='avc1'):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        map_size = config["map_size"]
        video_size = (8 * map_size, 4 * map_size)
        out = cv2.VideoWriter(output_file, fourcc, 5, video_size)

        frame_num = 0
        while True:
            map_img = self.step(0)
            if map_img is not None:
                height, width = map_img.shape[:2]
                map_img = cv2.resize(map_img, (2 * width, 2 * height))
                out.write(map_img)
            else:
                break

            frame_num += 1
            if frame_num >= max_frames:
                break

        out.release()

    def interactive(self):
        map_img = self.step(0)
        show_img(map_img)
        cv2.namedWindow('vis', cv2.WINDOW_NORMAL)
        while cv2.getWindowProperty('vis', cv2.WND_PROP_VISIBLE) >= 1:
            pressed_key = cv2.waitKey(10)
            if pressed_key == ord('j'):
                map_img = self.step(-1)
            elif pressed_key == ord('k'):
                map_img = self.step(1)

            if map_img is not None:
                show_img(map_img)
            else:
                break


def sample_waypoints(start_position, alpha, num_samples, alpha_range=120, min_delta=7, max_delta=12):
    alpha_noise = np.random.randint(alpha_range, size=num_samples) - alpha_range / 2
    alpha_rad = np.radians(alpha + alpha_noise)
    x_delta = np.random.randint(min_delta, max_delta, size=num_samples) * np.cos(alpha_rad)
    y_delta = np.random.randint(min_delta, max_delta, size=num_samples) * np.sin(alpha_rad)
    x_delta = x_delta.astype(int)
    y_delta = y_delta.astype(int)
    deltas = np.column_stack((x_delta, y_delta))
    candidate_waypoints = deltas + start_position
    return candidate_waypoints


def main(dataset_name, map_type, model_path, config_path, output_file, max_frames):
    config = get_config(config_path)

    maps = tracks_mapping(map_type)
    map_name = maps[dataset_name]
    map_reader = MapReader(Path(config["map_path"]) / map_name, config["map_size"])

    dataset = create_dataset(dataset_name, map_reader, config)
    simulator = MapSimulator(dataset, map_reader, model_path, config)

    if output_file:
        simulator.create_video(output_file, max_frames, config)
    else:
        simulator.interactive()


def show_img(map_img):
    height, width = map_img.shape[:2]
    map_img = cv2.resize(map_img, (2 * width, 2 * height))
    cv2.imshow('vis', map_img)


if __name__ == "__main__":
    args = parse_arguments()
    main(args.dataset_name, args.map_type, args.model_path, args.config_path,
         args.output_file, args.max_frames)
