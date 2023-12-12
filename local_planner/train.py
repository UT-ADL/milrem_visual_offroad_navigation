import argparse
import os
import time
from pathlib import Path

import torch
import wandb
import yaml

from lightning_fabric import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from gnm_train.data.gnm_dataset import GNMDataModule
from model.gnm import GNM, GNMExperiment
from model.util import convert_to_onnx, load_checkpoint
from model.vae import WaypointVAEGNM, WaypointVAEExperiment
from viz.visualizer import create_visualizer
from viz.visualizer import load_dataset as load_viz_dataset


def parse_arguments():
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--config",
        "-c",
        default="config/vae.yaml",
        type=str,
        help="Path to the config file in train_config folder",
    )

    argparser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        help="Path to model weight",
    )

    return argparser.parse_args()


def train_model():
    args = parse_arguments()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    with open("config/env.yaml", "r") as f:
        env_config = yaml.safe_load(f)
        config.update(env_config)

    seed_everything(config['seed'], True)

    config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    config["project_folder"] = os.path.join(
        "logs", config["project_name"], config["run_name"]
    )
    os.makedirs(
        config[
            "project_folder"
        ],  exist_ok=True
    )

    experiment = create_experiment(config["model_type"], config)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=experiment.device)
        load_checkpoint(experiment.model, ckpt)

    data_module = GNMDataModule(config)

    tb_logger = TensorBoardLogger(save_dir=config['save_dir'],
                                  name=config['project_name'], )

    wandb_logger = WandbLogger(project=config["project_name"], name=config["run_name"], log_model="all")
    model_checkpoint = ModelCheckpoint(save_top_k=5, dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                                 monitor="milrem_val_loss", save_last=True)

    runner = Trainer(logger=[tb_logger, wandb_logger],
                     callbacks=[
                         LearningRateMonitor(),
                         model_checkpoint,
                     ],
                     accelerator=config["accelerator"],
                     devices=config["devices"],
                     max_epochs=config["max_epochs"],
                     num_sanity_val_steps=config["num_sanity_val_steps"],
                     gradient_clip_val=1.0)
    print(f"======= Training {config['project_name']} =======")
    runner.fit(experiment, datamodule=data_module)

    best_model_name = "model-" + config["run_name"] + "-best"
    best_onnx_path = log_model(config, model_checkpoint.best_model_path, best_model_name, wandb_logger, ["best"])

    latest_model_name = "model-" + config["run_name"] + "-latest"
    log_model(config, model_checkpoint.last_model_path, latest_model_name, wandb_logger, ["latest"])

    if 'viz_video_dataset' in config:
        log_viz_video(config, best_onnx_path, use_sampled_waypoint=True)
        log_viz_video(config, best_onnx_path, use_sampled_waypoint=False)
    else:
        print("No 'viz_video_dataset' in configuration defined, skipping video creation.")


def log_viz_video(config, model_path, use_sampled_waypoint):
    viz = create_visualizer("onnx", use_sampled_waypoint)
    viz_dataset = load_viz_dataset(Path(config['viz_video_dataset']), config)
    name_suffix = "waypoint" if use_sampled_waypoint else "goal"
    video_name = f"{config['run_name']}-{name_suffix}"
    video_output_path = str(Path(config["project_folder"]) / f"{video_name}.mp4")
    viz.create_video(viz_dataset, model_path, video_output_path)
    wandb.log({video_name: wandb.Video(video_output_path)})


def log_model(config, model_checkpoint, model_name, wandb_logger, aliases):
    onnx_output_path = Path(config["project_folder"]) / f"{model_name}.onnx"
    convert_to_onnx(model_checkpoint, str(onnx_output_path), config)

    model_artifact = wandb.Artifact("model-" + wandb_logger.experiment.id + "-onnx", type='model')
    model_artifact.add_file(onnx_output_path)
    wandb_logger.experiment.log_artifact(model_artifact, aliases=aliases)
    return onnx_output_path


def create_experiment(model_type, config):
    if model_type == "vae":
        experiment = create_vae_experiment(config)
    elif model_type == "gnm":
        experiment = create_gnm_experiment(config)
    else:
        raise Exception(f"Unknown model type {model_type}")
    return experiment


def create_gnm_experiment(config):
    model = GNM(
        config["context_size"],
        config["len_traj_pred"],
        config["learn_angle"],
        config["obs_encoding_size"],
        config["goal_encoding_size"],
    )

    experiment = GNMExperiment(model,
                               float(config["alpha"]),
                               float(config["lr"]),
                               config["num_images_log"],
                               config["normalize"],
                               config["project_folder"])

    return experiment


def create_vae_experiment(config):
    model = WaypointVAEGNM(
        config["context_size"],
        config["obs_encoding_size"],
        config["goal_encoding_size"],
        config["latent_dim"],
        config["len_traj_pred"],
        config["learn_angle"])

    experiment = WaypointVAEExperiment(model,
                                       float(config["alpha"]),
                                       float(config["kld_weight"]),
                                       float(config["lr"]),
                                       config["num_images_log"],
                                       config["normalize"],
                                       config["project_folder"],
                                       config["scheduler_gamma"])

    return experiment


if __name__ == "__main__":
    train_model()
