import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from matplotlib import cm
from tqdm.auto import tqdm

from data.dataset import OneHotEncodedDataset, tracks_mapping
from data.mapping import MapReader
from models.util import load_model


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
    maps = tracks_mapping(config['map_type'])
    map_name = maps[dataset_name]

    map_reader = MapReader(Path(config["map_path"]) / map_name, config["map_size"])
    dataset = OneHotEncodedDataset(dataset_path=Path(config["dataset_path"]) / dataset_name,
                                   map_reader=map_reader,
                                   map_size=config["map_size"],
                                   # create trajectory with constant max length
                                   trajectory_min_len=config['trajectory_sim_length'],
                                   trajectory_max_len=config['trajectory_sim_length'],
                                   trajectory_sampling_rate=config['trajectory_sampling_rate'],
                                   config=config)

    return dataset


def create_video(model, dataset, output_file, config, codec='avc1'):
    fourcc = cv2.VideoWriter_fourcc(*codec)
    map_size = config["map_size"]
    video_size = (4 * map_size, 2 * map_size)
    out = cv2.VideoWriter(output_file, fourcc, 5, video_size)

    for i, batch in tqdm(enumerate(dataset), total=len(dataset)):
        masked_map, trajectory_mask, trajectory, map_img = batch
        masked_map = torch.tensor(masked_map).to("cuda")
        masked_map = masked_map.unsqueeze(dim=0)
        pred = model(masked_map).squeeze().detach().cpu().numpy()

        min_val = pred.min()
        max_val = pred.max()
        pred_norm = (pred - min_val) / (max_val - min_val)
        pred_norm = (256 * pred_norm).astype(np.uint8)

        pred_viridis = cm.viridis(pred_norm)
        pred_viridis = (pred_viridis * 255).astype(np.uint8)
        pred_viridis_bgr = cv2.cvtColor(pred_viridis, cv2.COLOR_RGBA2RGB)

        for t_idx in range(len(trajectory)):
            wp = trajectory[t_idx]
            wp_i = (int(wp[0]), int(wp[1]))
            cv2.circle(map_img, wp_i, 1, (0, 0, 255), 2)

        result_img = np.concatenate((map_img, pred_viridis_bgr), axis=1)
        result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)

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
