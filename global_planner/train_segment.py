import argparse
import os
import time
from pathlib import Path

import pytorch_lightning as pl
import torch
import wandb
import yaml
from lightning_fabric import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.cli import ReduceLROnPlateau
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from torch.nn import BCEWithLogitsLoss

from data.dataset import OneHotEncodedDataModule, tracks_mapping, val_maps
from data.mapping import MapReader
from models.binary_focal_loss import BinaryFocalLoss
from models.heuristic import SegmentiveHeuristic
from models.util import load_model, convert_to_onnx
from viz.create_segment_video import create_video, create_dataset
from viz.figure_util import create_segmentation_figure
from viz.simulator import MapSimulator


def parse_arguments():
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--run-name",
        "-r",
        type=str,
        required=False,
        help="Descriptive name for the experiment for logging and monitoring purposes.",
    )

    argparser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to configuration file.",
    )

    argparser.add_argument(
        "--sweep-id",
        type=str,
        required=False,
        help="W&B sweep id to run as sweep agent. If ommitted, run as training script.",
    )

    argparser.add_argument(
        "--sweep-config",
        type=str,
        required=False,
        help="W&B sweep id to run as sweep agent. If ommitted, run as training script.",
    )

    return argparser.parse_args()


class SegmentiveModule(pl.LightningModule):

    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.project_folder = Path(config["project_folder"])
        self.map_type = config["map_type"]
        self.map_size = config["map_size"]
        self.map_path = config["map_path"]
        self.dataset_path = config["dataset_path"]
        self.trajectory_min_length = config["trajectory_min_length"]
        self.trajectory_max_length = config["trajectory_max_length"]
        self.trajectory_sampling_rate = config["trajectory_sampling_rate"]
        self.lr = float(config["lr"])
        self.lr_patience = float(config["lr_patience"])
        self.lr_factor = float(config["lr_factor"])
        self.config = config

        if "loss" in config and config["loss"] == "binary_focal_loss":
            self.loss = BinaryFocalLoss(alpha=3, gamma=2)
        else:
            self.loss = BCEWithLogitsLoss()

        #self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        masked_map, trajectory_mask, trajectory, _ = batch
        predicted_prob = self.model(masked_map)
        loss = self.loss(predicted_prob.squeeze(), trajectory_mask)
        self.log("loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        masked_map, trajectory_mask, trajectory, _ = batch
        predicted_prob = self.model(masked_map)
        loss = self.loss(predicted_prob.squeeze(), trajectory_mask)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero:
            self.create_viz("2023-05-25-16-33-18", f"Kirikuküla", 28, "train")
            self.create_viz("2023-05-25-16-33-18", f"Kirikuküla", 746, "train")
            self.create_viz("2023-06-13-15-49-17", f"Piigaste", 163, "train")
            self.create_viz("2023-06-13-15-49-17", f"Piigaste", 544, "train")
            self.create_viz("2023-08-21-15-17-51", f"Ilmatsalu", 378, "train")
            self.create_viz("2023-08-21-15-17-51", f"Ilmatsalu", 711, "train")
            self.create_viz("2023-08-21-15-17-51", f"Ilmatsalu", 946, "train")
            self.create_viz("2023-05-11-17-08-21", f"Kärgandi", 81, "val")
            self.create_viz("2023-08-25-15-48-47", f"Ihaste", 588, "val")

    def create_viz(self, dataset, map_name, trajectory_index, dataset_type):
        val_viz_folder = self.project_folder / "visualize" / dataset_type
        val_viz_folder.mkdir(parents=True, exist_ok=True)
        val_save_path = val_viz_folder / f"{self.current_epoch}.png"

        map_path = Path(self.map_path) / f"{map_name}_{self.map_type}.tif"
        dataset_path = Path(self.dataset_path) / dataset

        create_segmentation_figure(self.model,
                                   dataset_path,
                                   str(map_path),
                                   map_size=self.map_size,
                                   trajectory_length=min(40, self.trajectory_max_length),
                                   trajectory_sampling_rate=self.trajectory_sampling_rate,
                                   trajectory_index=trajectory_index,
                                   save_path=val_save_path,
                                   config=self.config)

        wandb_logger = self.get_wandb_logger()
        if wandb_logger:
            wandb_logger.log_image(f"{dataset_type}_{map_name}_{trajectory_index}".lower(),
                                   [wandb.Image(str(val_save_path))])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, patience=self.lr_patience, factor=self.lr_factor,
                                               monitor="val_loss"),
                "monitor": "val_loss",
            },
        }

    def get_wandb_logger(self):
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                return logger
        return None


def read_config(config_path, run_name):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    with open("config/env.yaml", "r") as f:
        env_config = yaml.safe_load(f)
        config.update(env_config)

    seed_everything(config['seed'], True)

    if run_name:
        config["run_name"] += "_" + run_name
    config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    config["project_folder"] = os.path.join("logs", config["project_name"], config["run_name"])
    os.makedirs(config["project_folder"], exist_ok=True)
    return config


def train(config):
    wandb_logger = WandbLogger(project=config["project_name"], name=config["run_name"], log_model="all")
    tensorboard_logger = TensorBoardLogger(save_dir=config['save_dir'], name=config['project_name'])
    model_checkpoint = ModelCheckpoint(save_top_k=2,
                                       dirpath=os.path.join(tensorboard_logger.log_dir, "checkpoints"),
                                       monitor="val_loss", save_last=True)

    trainer = pl.Trainer(max_epochs=config["max_epochs"],
                         devices=config["trainer_devices"],
                         strategy=config["trainer_strategy"],
                         logger=[tensorboard_logger, wandb_logger],
                         callbacks=[LearningRateMonitor(), model_checkpoint])

    model = SegmentiveHeuristic(n_channels=5, n_classes=1)
    module = SegmentiveModule(model, config)

    dm = OneHotEncodedDataModule(map_size=config["map_size"],
                                 map_type=config["map_type"],
                                 map_path=config["map_path"],
                                 dataset_path=config["dataset_path"],
                                 num_workers=config["num_workers"],
                                 batch_size=config["batch_size"],
                                 trajectory_min_len=config["trajectory_min_length"],
                                 trajectory_max_len=config["trajectory_max_length"],
                                 trajectory_sampling_rate=config["trajectory_sampling_rate"],
                                 config=config)

    trainer.fit(module, datamodule=dm)

    if trainer.is_global_zero:
        best_model_name = "model-" + config["run_name"] + "-best"
        best_onnx_path = log_onnx_model(config, model_checkpoint.best_model_path, best_model_name, wandb_logger,
                                        ["best"])

        latest_model_name = "model-" + config["run_name"] + "-latest"
        log_onnx_model(config, model_checkpoint.last_model_path, latest_model_name, wandb_logger, ["latest"])

        for dataset_name in val_maps(config["map_type"]).keys():
            log_predictions_video(model_checkpoint.best_model_path, dataset_name, config)
            # Simulation video is create with ONNX model,
            # so all the final model used in deployment would be validated
            log_simulation_video(best_onnx_path, dataset_name, config)


def log_onnx_model(config, model_checkpoint, model_name, wandb_logger, aliases):
    onnx_output_path = Path(config["project_folder"]) / f"{model_name}.onnx"
    convert_to_onnx(model_checkpoint, str(onnx_output_path), config)

    model_artifact = wandb.Artifact("model-" + wandb_logger.experiment.id + "-onnx", type='model')
    model_artifact.add_file(onnx_output_path)
    wandb_logger.experiment.log_artifact(model_artifact, aliases=aliases)
    return onnx_output_path


def log_predictions_video(model_path, dataset_name, config):
    model = load_model(model_path)
    model = model.eval().to("cuda")
    dataset = create_dataset(dataset_name, config)

    map_name = tracks_mapping(config["map_type"])[dataset_name]
    map_name = Path(map_name).stem
    video_name = f"{dataset_name}-{map_name}-pred"

    video_output_path = Path(config["project_folder"]) / f"{video_name}.mp4"
    print(f"Creating prediction video for {dataset_name}...")
    create_video(model, dataset, str(video_output_path), config)

    wandb.log({video_name: wandb.Video(str(video_output_path))})


def log_simulation_video(model_path, dataset_name, config, max_frames=500):
    maps = tracks_mapping(config['map_type'])
    map_name = maps[dataset_name]
    map_reader = MapReader(Path(config["map_path"]) / map_name, config["map_size"])

    dataset = create_dataset(dataset_name, config)
    simulator = MapSimulator(dataset, map_reader, model_path, config)

    video_name = f"{dataset_name}-{Path(map_name).stem}-sim"
    video_output_path = Path(config["project_folder"]) / f"{video_name}.mp4"
    print(f"Creating simulation video for {dataset_name}...")
    simulator.create_video(str(video_output_path), max_frames, config)

    wandb.log({video_name: wandb.Video(str(video_output_path))})


if __name__ == "__main__":
    args = parse_arguments()
    config = read_config(args.config, args.run_name)

    if args.sweep_id:
        print(f"Training as sweep {args.sweep_id} agent")
        def train_lambda(): train(config)
        wandb.agent(sweep_id=args.sweep_id, function=train_lambda)

    elif args.sweep_config:
        print(f"Running sweep {args.sweep_config}")
        def train_lambda(): train(config)
        with open(args.sweep_config, "r") as f:
            sweep_config = yaml.safe_load(f)

        sweep_id = wandb.sweep(sweep_config, project=config["project_name"])
        wandb.agent(sweep_id=sweep_id, function=train_lambda)

    else:
        train(config)
