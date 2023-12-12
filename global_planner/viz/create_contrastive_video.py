import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from matplotlib import cm
from tqdm.auto import tqdm

from data.dataset import tracks_mapping, TrajectoryDataset
from data.mapping import MapReader
from models.heuristic import ContrastiveHeuristic


def parse_arguments():
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to configuration file.",
    )

    argparser.add_argument(
        "--dataset-name",
        "-d",
        type=str,
        required=True,
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
        "--output-file",
        "-o",
        type=str,
        required=True,
        help="Path to model checkpoint.",
    )

    return argparser.parse_args()


def get_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    with open("config/env.yaml", "r") as f:
        env_config = yaml.safe_load(f)
        config.update(env_config)

    return config


def create_dataset(dataset_name, config):
    maps = tracks_mapping(config["map_type"])
    map_name = maps[dataset_name]
    map_reader = MapReader(Path(config["map_path"]) / map_name, config["map_size"])

    dataset = TrajectoryDataset(Path(config["dataset_path"]) / dataset_name,
                                map_reader,
                                map_size=config["map_size"],
                                trajectory_min_len=config['trajectory_max_length'],
                                trajectory_max_len=config['trajectory_max_length'],
                                trajectory_sampling_rate=config['trajectory_sampling_rate']
                                )

    return dataset


def load_model(model_path):
    model = ContrastiveHeuristic()
    ckpt = torch.load(model_path, map_location="cuda:0")
    state_dict = ckpt['state_dict']
    state_dict = {re.sub(r'^model\.', '', k): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def create_video(model, dataset, output_file, config):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    map_size = config["map_size"]
    video_size = (2 * map_size, 2 * map_size)
    out = cv2.VideoWriter(output_file, fourcc, 5, video_size)

    denorm = lambda x: x * map_size + map_size

    for i, batch in tqdm(enumerate(dataset), total=len(dataset)):

        with torch.no_grad():
            trajectory, map_tensor, map_img = batch
            map_tensor = map_tensor.unsqueeze(dim=0)
            map_tensor = map_tensor.to("cuda")
            map_features = model.encode_map(map_tensor)

            t = torch.linspace(-1.0, 1.0, 60)
            x, y = torch.meshgrid(t, t)
            wps = torch.stack((x, y), dim=-1)

            neg_probs = np.zeros((wps.shape[0], wps.shape[1]))
            for i in range(wps.shape[0]):
                for j in range(wps.shape[1]):
                    neg_coord = torch.cat([trajectory[0], trajectory[-1], wps[i, j]]).unsqueeze(dim=0).to("cuda")
                    neg_probs[i, j] = model(neg_coord, map_features).item()

            wps_denorm = denorm(wps).view(-1, 2)
            neg_probs = neg_probs.flatten()
            neg_probs_filter = np.where(neg_probs > 0.01)
            neg_probs = neg_probs[neg_probs_filter]
            wps_denorm = wps_denorm[neg_probs_filter]

            neg_probs = (256 * neg_probs).astype(np.uint8)
            neg_probs = cm.Reds(neg_probs)
            neg_probs = (256 * neg_probs).astype(np.uint8)

            for w_idx in range(len(wps_denorm)):
                wp = wps_denorm[w_idx]
                prob = neg_probs[w_idx]
                wp_i = (int(wp[0]), int(wp[1]))
                cv2.circle(map_img, wp_i, 1, (int(prob[0]), int(prob[1]), int(prob[2])), 2)

            for t_idx in range(len(trajectory)):
                wp = denorm(trajectory[t_idx])
                wp_i = (int(wp[0]), int(wp[1]))
                cv2.circle(map_img, wp_i, 1, (0, 0, 255), 2)

        result_img = cv2.cvtColor(map_img, cv2.COLOR_RGB2BGR)
        out.write(result_img)

    out.release()


def main(config_path, dataset_name, model_path, output_file):
    config = get_config(config_path)
    dataset = create_dataset(dataset_name, config)
    model = load_model(model_path)
    model = model.to("cuda")
    create_video(model, dataset, output_file, config)


if __name__ == "__main__":
    args = parse_arguments()
    main(args.config, args.dataset_name, args.model_path, args.output_file)
