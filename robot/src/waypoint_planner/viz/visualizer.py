import argparse
import os
import shutil
from abc import abstractmethod
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from tqdm import tqdm

from data.dataset import MilremVizDataset
from gnm_train.models import mdn
from model.util import load_model
from viz.util_camera import load_camera_model, to_camera_frame

import onnxruntime as ort

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
FONT_SIZE = 0.3


def parse_arguments():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--dataset-path',
        required=True
    )
    argparser.add_argument(
        '--model-config',
        required=True
    )
    argparser.add_argument(
        '--model-path',
        required=True
    )
    argparser.add_argument(
        '--onnx',
        required=False,
        action='store_true',
        help="Whether to use model is in onnx format instead of PyTorch."
    )
    argparser.add_argument(
        '--output-file',
        '-o',
        required=False,
        help="Output video file name."
    )
    argparser.add_argument(
        '--start-frame',
        required=False,
        default=0,
        type=int,
        help="Dataset location to start from creating the video"
    )
    argparser.add_argument(
        '--use-sampled-waypoint',
        required=False,
        action='store_true',
        help="Whether to use goal image from sampled waypoint or from last frame of the track."
    )
    return argparser.parse_args()


class Visualizer:
    TOP_DOWN_VIEW_SCALE = 15
    
    def __init__(self, use_sampled_waypoint=False):
        self.camera_model = load_camera_model()
        self.use_sampled_waypoint = use_sampled_waypoint

    @classmethod
    def create(cls, model_type, use_sampled_waypoint):
        if model_type == "vae":
            return VAEVisualiser(use_sampled_waypoint)
        elif model_type == "mdn":
            return MDNVisualiser(use_sampled_waypoint)
        elif model_type in ["gnm", "gnm-pretrained"]:
            return GNMVisualiser(use_sampled_waypoint)
        elif model_type == "onnx":
            return OnnxVisualiser(use_sampled_waypoint)
        else:
            raise Exception(f"Unknown model type: {model_type}")

    @abstractmethod
    def create_frame(self, model, dataset, frame_index):
        """

        Create visualisation frame with predictions overlayed on top of it.
        Overwrite this in model specific class by reading data from dataset, make prediction using the model
        and overlay it on top of frame.

        :param model: Model used to make predictions
        :param dataset: Dataset used to read data
        :param frame_index: Frame index, same as dataset index
        :return: image frame
        """
        pass

    def draw_info_overlay(self, frame, labels, data, predictions):

        self.draw_rectangle(frame, 5, 15, 90, 130)

        cv2.putText(frame, 'cur frame: {}'.format(data['idx']), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, GREEN, 1, cv2.LINE_AA)
        cv2.putText(frame, 'wp frame: {}'.format(data['wp_idx']), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, GREEN, 1, cv2.LINE_AA)

        cv2.putText(frame, 'cur x: {:.3f}'.format(data['cur_pos_x']), (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, GREEN, 1, cv2.LINE_AA)
        cv2.putText(frame, 'cur y: {:.3f}'.format(data['cur_pos_y']), (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, GREEN, 1, cv2.LINE_AA)

        cv2.putText(frame, 'true dist: {:.3f}'.format(labels[1][0].item()), (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, GREEN, 1, cv2.LINE_AA)

        if predictions[0]:
            cv2.putText(frame, 'pred dist: {:.3f}'.format(predictions[0][0].item()), (10, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, RED, 1, cv2.LINE_AA)

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

    def convert_frames_to_video(self, frames_folder, output_video_path, fps=15):
        output_folder = Path(os.path.split(output_video_path)[:-1][0])
        output_folder.mkdir(parents=True, exist_ok=True)

        p = Path(frames_folder).glob('**/*.jpg')
        image_list = sorted([str(x) for x in p if x.is_file()])

        print("Creating video {}, FPS={}".format(frames_folder, fps))
        clip = ImageSequenceClip(image_list, fps=fps)
        clip.write_videofile(str(output_video_path))

    def create_video(self, dataset, model, output_file_name, start_frame=0):
        temp_frames_folder = Path('./temp')
        shutil.rmtree(temp_frames_folder, ignore_errors=True)
        temp_frames_folder.mkdir()

        for frame_index in tqdm(range(start_frame, len(dataset))):
            img = self.create_frame(model, dataset, frame_index)
            cv2.imwrite(f"{temp_frames_folder}/{frame_index + 1:05}.jpg", img)

        self.convert_frames_to_video(temp_frames_folder, output_file_name)
        shutil.rmtree(temp_frames_folder, ignore_errors=True)

    def create_interactive(self, dataset, model, start_frame):
        deq = deque(range(0, len(dataset)))
        deq.rotate(-start_frame)
        cv2.namedWindow('vis', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('vis', 2 * 640, 2 * 360)
        self.show_next_frame(model, dataset, deq)
        while cv2.getWindowProperty('vis', cv2.WND_PROP_VISIBLE) >= 1:
            pressed_key = cv2.waitKey(10)
            if pressed_key == ord('j'):
                deq.rotate(1)
                self.show_next_frame(model, dataset, deq)
            elif pressed_key == ord('k'):
                deq.rotate(-1)
                self.show_next_frame(model, dataset, deq)
            if pressed_key == ord('u'):
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


class VAEVisualiser(Visualizer):

    def create_frame(self, model, dataset, frame_index):
        obs_tensor, waypoint_tensor, labels, data, obs_img, waypoint_img = dataset[frame_index]

        if not self.use_sampled_waypoint:
            last_obs_tensor, _, _, last_data, waypoint_img, _ = dataset[len(dataset) - 1]
            waypoint_tensor = last_obs_tensor[-3:]
            data["wp_idx"] = last_data["idx"]

        rectified_image = self.rectify_image(obs_img)

        with torch.no_grad():
            predictions, _, _, _ = model(obs_tensor.unsqueeze(dim=0), waypoint_tensor.unsqueeze(dim=0))
            distance, actions = model.sample(obs_tensor.unsqueeze(dim=0), 20)
            waypoint_samples = actions.detach().numpy()

        predicted_actions = predictions[1].squeeze().detach().numpy()

        if waypoint_samples is not None:
            for i in range(waypoint_samples.shape[0]):
                sampled_trajectory = waypoint_samples[i]
                to_camera_frame(rectified_image, sampled_trajectory, BLUE)

        to_camera_frame(rectified_image, labels[0], GREEN)
        to_camera_frame(rectified_image, predicted_actions, RED)

        self.draw_info_overlay(rectified_image, labels, data, predictions)
        self.draw_top_town_overlay(rectified_image, waypoint_samples, predicted_actions, labels[0])
        self.draw_waypoint_img(rectified_image, waypoint_img)

        return rectified_image


class OnnxVisualiser(Visualizer):

    def create_frame(self, model, dataset, frame_index):
        obs_tensor, waypoint_tensor, labels, data, obs_img, waypoint_img = dataset[frame_index]

        if not self.use_sampled_waypoint:
            last_obs_tensor, _, _, last_data, waypoint_img, _ = dataset[len(dataset) - 1]
            waypoint_tensor = last_obs_tensor[-3:]
            data["wp_idx"] = last_data["idx"]

        rectified_image = self.rectify_image(obs_img)

        forward_session = ort.InferenceSession(
            model,
            providers=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"]
        )

        forward_model_in = {'obs_img': obs_tensor.unsqueeze(dim=0).numpy(),
                            'goal_img': waypoint_tensor.unsqueeze(dim=0).numpy()}
        forward_model_out = forward_session.run(None, forward_model_in)

        predicted_distances = forward_model_out[0]
        predicted_actions = np.squeeze(forward_model_out[1])
        predicted_actions = predicted_actions[:, :2]  # remove angle

        if len(forward_model_out) >= 4:
            waypoint_samples = np.squeeze(forward_model_out[3])

            for i in range(waypoint_samples.shape[0]):
                sampled_trajectory = waypoint_samples[i]
                to_camera_frame(rectified_image, sampled_trajectory, BLUE)
        else:
            waypoint_samples = None

        to_camera_frame(rectified_image, labels[0], GREEN)
        to_camera_frame(rectified_image, predicted_actions, RED)

        self.draw_info_overlay(rectified_image, labels, data, (predicted_distances, predicted_actions))
        self.draw_top_town_overlay(rectified_image, waypoint_samples, predicted_actions, labels[0])
        self.draw_waypoint_img(rectified_image, waypoint_img)

        return rectified_image


class MDNVisualiser(Visualizer):

    def create_frame(self, model, dataset, frame_index):
        obs_tensor, waypoint_tensor, labels, data, obs_img, waypoint_img = dataset[frame_index]

        if not self.use_sampled_waypoint:
            last_obs_tensor, _, _, last_data, waypoint_img, _ = dataset[len(dataset) - 1]
            waypoint_tensor = last_obs_tensor[-3:]
            data["wp_idx"] = last_data["idx"]

        rectified_image = self.rectify_image(obs_img)

        with torch.no_grad():
            dist_pred, action_pred = model(obs_tensor.unsqueeze(dim=0), waypoint_tensor.unsqueeze(dim=0))
            dist_pi, dist_sigma, dist_mu = dist_pred
            action_pi, action_sigma, action_mu = action_pred

            predicted_actions = torch.cumsum(action_mu, dim=2).squeeze()

            waypoint_samples = []
            for i in range(20):
                flat_sigma = action_sigma.reshape(action_sigma.shape[0], action_sigma.shape[1], -1)
                flat_mu = action_mu.reshape(action_mu.shape[0], action_mu.shape[1], -1)
                waypoint_sample = mdn.sample(action_pi, flat_sigma, flat_mu)
                waypoint_sample = waypoint_sample.squeeze().reshape(5, 2)
                waypoint_sample = np.cumsum(waypoint_sample, axis=0)
                to_camera_frame(rectified_image, waypoint_sample, BLUE)
                waypoint_samples.append(waypoint_sample)

            to_camera_frame(rectified_image, labels[0], GREEN)
            for i in range(predicted_actions.shape[0]):
                to_camera_frame(rectified_image, predicted_actions[i], RED)

        self.draw_info_overlay(rectified_image, labels, data, ([dist_mu.mean()], predicted_actions))
        self.draw_top_town_overlay(rectified_image, np.array(waypoint_samples), predicted_actions.squeeze(), labels[0])
        self.draw_waypoint_img(rectified_image, waypoint_img)

        return rectified_image


class GNMVisualiser(Visualizer):

    def create_frame(self, model, dataset, frame_index):
        obs_tensor, waypoint_tensor, labels, data, obs_img, waypoint_img = dataset[frame_index]
        rectified_image = self.rectify_image(obs_img)

        if not self.use_sampled_waypoint:
            last_obs_tensor, _, _, last_data, waypoint_img, _ = dataset[len(dataset) - 1]
            waypoint_tensor = last_obs_tensor[-3:]
            data["wp_idx"] = last_data["idx"]

        with torch.no_grad():
            predictions = model(obs_tensor.unsqueeze(dim=0), waypoint_tensor.unsqueeze(dim=0))

        predicted_actions = predictions[1].squeeze().detach().numpy()

        to_camera_frame(rectified_image, labels[0], GREEN)
        to_camera_frame(rectified_image, predicted_actions, RED)
        self.draw_info_overlay(rectified_image, labels, data, predictions)
        self.draw_top_town_overlay(rectified_image, None, predictions[1][0].squeeze(), labels[0])
        self.draw_waypoint_img(rectified_image, waypoint_img)

        return rectified_image


def load_dataset(dataset_path, config):
    return MilremVizDataset(dataset_path, **config)


if __name__ == "__main__":

    args = parse_arguments()
    dataset_path = Path(args.dataset_path)
    model_config = args.model_config
    model_path = args.model_path
    is_onnx = args.onnx
    output_file = args.output_file
    start_frame = args.start_frame
    use_sampled_waypoint = args.use_sampled_waypoint

    with open(model_config, 'r') as file:
        config = yaml.safe_load(file)

    if is_onnx:
        # There is no model to load, onnx inference session is created inside the visualizer
        model = model_path
        model_type = "onnx"
    else:
        model = load_model(model_path, config)
        model.eval()
        model.train(False)
        model_type = config['model_type']

    dataset = load_dataset(dataset_path, config)
    viz = Visualizer.create(model_type, use_sampled_waypoint)

    if output_file:
        viz.create_video(dataset, model, output_file, start_frame)
    else:
        viz.create_interactive(dataset, model, start_frame)
