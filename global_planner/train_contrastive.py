import argparse
import os
import time
from pathlib import Path

import torch
import wandb
import yaml
from lightning_fabric import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.cli import ReduceLROnPlateau
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from data.dataset import TrajectoryDataModule
from models.heuristic import ContrastiveHeuristic

import pytorch_lightning as pl

from viz.figure_util import create_contrastive_figure


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

    return argparser.parse_args()


class ContrastiveModule(pl.LightningModule):

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

    def training_step(self, batch, batch_idx):
        neg_prob, pos_prob = self.predict_probabilities(batch)
        infoNCE_loss = self.calculate_infonce_loss(neg_prob, pos_prob)
        self.log("pos_prob", pos_prob.mean(), prog_bar=True)
        self.log("neg_prob", neg_prob.mean(), prog_bar=True)
        self.log("loss", infoNCE_loss, prog_bar=True, sync_dist=True)
        return infoNCE_loss

    def validation_step(self, batch, batch_idx):
        neg_prob, pos_prob = self.predict_probabilities(batch)
        infoNCE_loss = self.calculate_infonce_loss(neg_prob, pos_prob)

        self.log("val_pos_prob", pos_prob.mean(), prog_bar=True)
        self.log("val_neg_prob", neg_prob.mean(), prog_bar=True)
        self.log("val_loss", infoNCE_loss, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):
        self.create_viz("2023-05-25-16-33-18", f"Kirikuküla", 330, "train")
        self.create_viz("2023-05-25-16-33-18", f"Kirikuküla", 177, "train")
        self.create_viz("2023-06-13-15-49-17", f"Piigaste", 182, "train")
        self.create_viz("2023-06-13-15-49-17", f"Piigaste", 269, "train")
        self.create_viz("2023-05-11-17-08-21", f"Kärgandi", 30, "val")
        self.create_viz("2023-08-23-15-26-38", f"Ihaste", 50, "val")

    def create_viz(self, dataset, map_name, trajectory_index, dataset_type):
        val_viz_folder = self.project_folder / "visualize" / dataset_type
        val_viz_folder.mkdir(parents=True, exist_ok=True)
        val_save_path = val_viz_folder / f"{self.current_epoch}.png"

        map_path = Path(self.map_path) / f"{map_name}_{self.map_type}.tif"
        dataset_path = Path(self.dataset_path) / dataset

        create_contrastive_figure(self.model,
                                  dataset_path,
                                  str(map_path),
                                  map_size=self.map_size,
                                  trajectory_length=self.trajectory_max_length,
                                  trajectory_sampling_rate=self.trajectory_sampling_rate,
                                  trajectory_index=trajectory_index,
                                  save_path=val_save_path)
        wandb.log({f"{dataset_type}_{map_name}_{trajectory_index}".lower(): wandb.Image(str(val_save_path))})

    def predict_probabilities(self, batch):
        positive_coordinates, negative_coordinates, maps = batch
        map_features = self.model.encode_map(maps)

        n_pos_samples = positive_coordinates.shape[1]
        num_coordinates = positive_coordinates.shape[-1]
        positive_coordinates = positive_coordinates.view(-1, num_coordinates)
        positive_map_features = map_features.repeat(1, n_pos_samples).view(-1, map_features.shape[1])
        pos_prob = self.model(positive_coordinates, positive_map_features)

        n_neg_samples = negative_coordinates.shape[1]
        negative_coordinates = negative_coordinates.view(-1, num_coordinates)
        negative_map_features = map_features.repeat(1, n_neg_samples).view(-1, map_features.shape[1])
        neg_prob = self.model(negative_coordinates, negative_map_features)

        return neg_prob, pos_prob

    def calculate_infonce_loss(self, neg_results, pos_results):
        numerator = torch.exp(pos_results)
        # denominator = numerator + torch.sum(torch.exp(neg_results), dim=1, keepdim=True)
        denominator = numerator + torch.sum(torch.exp(neg_results))
        infoNCE_loss = -torch.log(numerator / denominator)
        infoNCE_loss = torch.mean(infoNCE_loss)
        return infoNCE_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, patience=self.lr_patience, factor=self.lr_factor, monitor="val_loss"),
                "monitor": "val_loss",
            },
        }


def read_config(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    with open("config/env.yaml", "r") as f:
        env_config = yaml.safe_load(f)
        config.update(env_config)

    seed_everything(config['seed'], True)
    if args.run_name:
        config["run_name"] += "_" + args.run_name
    config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    config["project_folder"] = os.path.join("logs", config["project_name"], config["run_name"])
    os.makedirs(config["project_folder"], exist_ok=True)
    return config


def train():
    args = parse_arguments()
    config = read_config(args)

    model = ContrastiveHeuristic()
    module = ContrastiveModule(model, config)

    dm = TrajectoryDataModule(map_size=config["map_size"],
                              map_type=config["map_type"],
                              map_path=config["map_path"],
                              dataset_path=config["dataset_path"],
                              num_workers=config["num_workers"],
                              batch_size=config["batch_size"],
                              trajectory_min_len=config["trajectory_min_length"],
                              trajectory_max_len=config["trajectory_max_length"],
                              trajectory_sampling_rate=config["trajectory_sampling_rate"],
                              n_negative_samples=config["num_negative_samples"],
                              negative_dist_thres=config["negative_distance_thres"])

    wandb_logger = WandbLogger(project=config["project_name"], name=config["run_name"], log_model="all")
    tensorboard_logger = TensorBoardLogger(save_dir=config['save_dir'], name=config['project_name'])
    model_checkpoint = ModelCheckpoint(save_top_k=5, dirpath=os.path.join(tensorboard_logger.log_dir, "checkpoints"),
                                       monitor="val_loss", save_last=True)

    trainer = pl.Trainer(max_epochs=config["max_epochs"],
                         logger=[tensorboard_logger, wandb_logger],
                         callbacks=[LearningRateMonitor(), model_checkpoint])
    trainer.fit(module, datamodule=dm)


if __name__ == "__main__":
    train()
